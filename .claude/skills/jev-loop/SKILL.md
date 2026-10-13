---
name: jev-loop
description: jev-gate 레포의 개발 검증 루프. 코드나 config를 바꾼 뒤 정적 검사 → 데이터 계약 → smoke E2E → 지표 회귀 → 사람 육안 → PR 기록 순서로 검증하고, 실패하면 필요한 단계부터 다시 돈다. 실행 전에 config 변경 의도를 묻는다. "/jev-loop", "검증 돌려줘", "PR 올리기 전에 확인", 실험 결과를 main에 합치기 전에 사용한다.
---

# jev-loop

이 레포(`jev-gate`)에서 코드나 config를 바꾼 뒤 main에 합치기 전에 반드시 이 루프를 돈다.
실험 설계와 공정성 규칙의 근거는 `docs/PLAN.md`에 있다. 이 문서와 충돌하면 PLAN.md가 우선한다.

```
⓪ 변경 의도 확인 → ① 정적 검사 → ② 데이터 계약 → ③ smoke E2E → ④ 지표 회귀 → ⑤ 사람 육안 → ⑥ PR 기록
                        ↑                                                          │
                        └──────────── 실패 시 아래 "되돌아갈 단계" 표대로 ───────────┘
```

## 공통 규칙

- 모든 명령은 레포 루트에서 `uv run ...`으로 실행한다 (`.venv`, torch cu128).
- `.env`의 `TYPESAFE_API_KEY`는 출력·로그·커밋·PR 본문 어디에도 쓰지 않는다. 키가 없으면 ③을 멈추고 사용자에게 알린다.
- **판정 기준을 통과하지 못한 단계를 통과로 보고하지 않는다.** 스크립트가 아직 없으면(`scripts/*.py` 미구현) 그 단계를 "미구현"으로 표시한다. 건너뛰고 다음 단계로 가도 되지만 ⑥ 보고서에 그대로 적는다.
- test 클립 결과를 보고 임계값·프롬프트·클립·모델을 바꾸지 않는다. 바꿔야 하면 dev에서 다시 정한다 (PLAN.md §6).
- 단계 결과는 매번 한 줄로 사용자에게 알린다: `② 데이터 계약: 통과 (candidates 1,204 / labels 1,204)`.

## ⓪ 변경 의도 확인 (실행 전)

```bash
uv run python scripts/config_guard.py --base main
```
`main` 대비 `configs/` diff를 읽어 아래 키가 바뀌었으면 **실행 전에 사용자에게 묻고 답을 기다린다**. 질문은 AskUserQuestion으로 한 번에 묶는다.

| 바뀐 키 | 물어볼 것 | 답에 따른 조치 |
|---|---|---|
| `thresholds.*`, `gate.tau`, `gate.max_auto_error`, `baseline.t` | "test 클립을 보고 바꾼 건가요?" | 그렇다 → 변경을 되돌리고 `fit_thresholds.py --split dev`로 다시 정함 |
| `clips.*` | "결과를 보기 전에 정한 분할인가요?" | 아니다 → 원래 분할로 되돌림 |
| `vlm.model`, `vlm.dtype` | "VLM이 바뀌었으니 ② 기준선을 다시 돌려야 합니다. 진행할까요?" | 예 → ③ smoke 후 ② 전체 재실행 계획을 ⑥에 기록 |
| `jev.model` (또는 응답의 `model` 버전 변경) | "Jev 버전이 바뀌었습니다(이전 → 현재). ③·④를 다시 돌릴까요?" | 예 → ③·④ 재실행 |
| `prompts.question` | "VLM과 Jev 둘 다 같은 문장으로 바뀌었나요?" | 한쪽만이면 중단 |
| `detector.*` | "후보 박스가 바뀌므로 네 시스템 전부 다시 돌립니다." | 전부 재실행 |

바뀐 키가 위 표에 없으면 묻지 않고 ①로 간다.

## ① 정적 검사

```bash
uv run ruff check .
uv run pytest -q
```
- **통과 기준:** ruff 오류 0, 테스트 전부 통과 (skip은 개수를 보고).
- 꼭 있어야 하는 테스트: IoU·NMS 계산, IoU 구간별 라벨(≥0.5 / <0.3 / 그 사이), ECE·Brier, 게이트 라우팅, evidence가 미래 프레임을 쓰지 않는지, config 상속·검증, Jev 응답 검증(`noul` ∈ [0,1]).
- 실패 → 코드를 고치고 ①부터.

## ② 데이터 계약

```bash
uv run python scripts/check_contracts.py --run-dir runs/<run_id>
```
가장 최근 run(없으면 ③을 먼저 돌린 뒤 돌아온다)의 단계별 JSONL을 PLAN.md §3 표대로 검사한다.
- **통과 기준:** 필수 필드 누락 0, 클립별 프레임 수 = 이미지 수, cand_id가 candidates·labels·모든 decisions에서 1:1, 박스가 프레임 안, 확률 ∈ [0,1], `routed_to=vlm`인 행이 ② 판정과 일치.
- evidence의 미래 누설 검사: 모든 트랙 특징이 `frame_idx` 이하의 후보만 참조.

## ③ smoke E2E

```bash
uv run python scripts/run_pipeline.py --config configs/smoke.yaml
```
`smoke.yaml`은 dev 클립 5프레임, 네 시스템 전부, `budget.max_jev_requests`와 `budget.max_vlm_calls`에 상한이 걸려 있다.
- **통과 기준:** S1~S7이 끝까지 실행되고, `metrics.json`이 생기고, Jev 요청 수와 토큰이 상한 이하. 실행 시간과 Jev 토큰 사용량을 보고한다.
- 상한을 넘으면 파이프라인이 스스로 멈춰야 한다. 멈추지 않았다면 버그로 보고 ①로 돌아간다.

## ④ 지표 회귀

```bash
uv run python scripts/run_pipeline.py --config configs/base.yaml --split dev
uv run python scripts/compare_metrics.py --run-dir runs/<run_id> --history metrics_history.jsonl
```
- dev 클립 전체를 돌리고 직전 기록과 비교한다. 지표 순서는 항상 정확도 → Precision → Recall → ECE → VLM 호출 → 지연 → 비용.
- **통과 기준:** 정확도·Precision·Recall이 0.01 넘게 떨어지거나 ECE가 0.01 넘게 오른 항목이 없다. 떨어졌다면 그 원인이 이번 변경으로 **설명되는** 경우만 통과로 본다(예: ambiguous 구간 정책 변경). 설명은 ⑥에 쓴다.
- 통과하면 결과를 `metrics_history.jsonl`에 한 줄 추가한다 (`run_id`, `git_commit`, `config_hash`, 지표).
- **test 클립은 이 단계에서 돌리지 않는다.** test는 M4에서 `thresholds.yaml`을 고정한 뒤 한 번만 돌린다.

## ⑤ 사람 육안 (사람이 직접)

```bash
uv run python scripts/render_samples.py --run-dir runs/<run_id> --n 10 --seed 0
```
- 무작위 10프레임에 후보 박스, y, 네 시스템의 판정을 겹친 이미지와 Before → After 영상을 만든다.
- 경로를 사용자에게 알리고 **확인을 기다린다.** Claude가 대신 통과 판정을 내리지 않는다.
- **통과 기준 (사용자 판단):** 박스와 판정이 화면과 맞는다. GT가 틀려 보이는 사례(흐릿한 실루엣 등)는 `runs/<run_id>/review_notes.md`에 따로 기록한다.

## ⑥ PR 기록

`.github/pull_request_template.md` 형식으로 PR 본문 초안을 만들어 채팅에 보여준다. 커밋·푸시·PR 생성은 사용자가 요청할 때만 한다.

```markdown
## 변경 요약
- 무엇을 왜 바꿨는지 2~3줄

## 지표 (dev, before → after)
| 시스템 | 정확도 | Precision | Recall | ECE | VLM 호출 | 지연(중앙값) | 비용 |
|---|---|---|---|---|---|---|---|
| ① detector | | | | – | 0 | | 0 |
| ② DINO+VLM | | | | – | | | |
| ③ DINO+Jev | | | | | 0 | | |
| ④ Jev 게이트 | | | | | | | |
n_box / n_frame / n_clip, DINO 후보 recall 상한 x%

## 검증 결과
⓪ 변경 의도 · ① 정적 검사 · ② 데이터 계약 · ③ smoke · ④ 회귀 · ⑤ 육안 — 단계별 통과/실패/미구현

## 재현
run_id, git commit, config 경로, Jev 모델 버전, VLM 모델
```

## 되돌아갈 단계

| 실패한 곳 | 원인 | 다시 시작할 단계 |
|---|---|---|
| ① | 코드 오류 | ① |
| ② 스키마·개수 | 해당 단계 코드 | ① (고친 뒤 ③에서 run 새로 생성) |
| ② 미래 누설 | `evidence.py` | ① |
| ③ 실행 중단 | 코드·환경 | ① |
| ③ 비용 상한 초과 | 배치 설정·상한 미작동 | ⓪ (config 확인) → ① |
| ④ 설명 안 되는 하락 | 이번 변경 | 변경을 다시 보고 ① |
| ⑤ 판정이 화면과 다름 | 라벨링·시각화·judge | 원인 모듈을 고친 뒤 ① |
| ⑤ GT가 틀림 | 데이터 | 코드 변경 없음. `review_notes.md`에 기록하고 ⑥에 각주 |

루프는 ⑥까지 모두 통과하거나, 사용자가 멈추라고 할 때 끝난다. 끝나면 단계별 결과를 한 줄씩 요약한다.
