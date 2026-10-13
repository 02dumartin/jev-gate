# jev-gate 실험 계획 (v1, 2026-10-13)

> 질문: Grounding DINO가 낸 후보 박스가 **진짜 물체인가**를 판정할 때,
> Jev(calibrated 확률)를 게이트로 쓰면 VLM 호출을 줄이면서 정확도를 유지할 수 있는가?

---

## 1. 비교할 시스템 (표의 행)

모든 시스템은 **같은 DINO 후보 박스**를 받아 박스마다 `keep / reject`를 낸다. 앞단은 고정하고 판정 단계만 비교한다.

| # | 시스템 | 판정 방식 | 확률 출력 | 정할 임계값 |
|---|---|---|---|---|
| ① | 기존 detector | DINO score ≥ `t` | 없음 | `t` (dev에서 정확도 최대) |
| ② | DINO + VLM | Qwen3-VL-8B가 모든 박스를 yes/no로 판정 | 없음 (p_yes는 부록용으로만 기록) | 없음 |
| ③ | DINO + Jev | Jev Noul `p_real ≥ 0.5` | 있음 | 없음 (0.5 고정: calibration을 믿는 설정) |
| ④ | Jev 게이트 + VLM | `max(p, 1−p) ≥ τ`면 Jev 판정, 아니면 VLM으로 넘김 | 자동 처리분만 | `τ` (dev에서 정함) |

- ④는 ②와 ③의 로그를 조합해서 계산한다. 추가 실행이 필요 없다.
- **Jev는 텍스트만 받는다** (TypeSafe 문서: "Jev accepts text only"). 그래서 ③은 픽셀이 아니라 **구조화된 증거**(§4)를 보고 판정한다. 발표에서 이 차이를 반드시 밝힌다.
- (선택, P2) ③′: 같은 텍스트 증거를 Qwen3-VL에 이미지 없이 넣는다. ②와 ③의 차이가 모델에서 오는지 입력에서 오는지 구분하는 대조군이다.

## 2. 데이터와 정답

### 2.1 정답 소스
- **AAU CSV** (`annotations/annotations_AAU/{train,valid,test}.csv`)를 GT로 쓴다. 12,444장, 박스 35,565개.
  - COCO json에는 valid 분할이 없고(11,206장), YOLO는 빈 파일 1,449장에 실제 박스가 누락돼 있다.
- AAU에 없는 2,640장은 YOLO도 전부 비어 있다 → `gt_status = empty_by_absence` (물체 없는 프레임).
  - 기본은 평가에 포함한다. 민감도 분석으로 제외한 결과도 같이 낸다.
- 분할 목록 밖의 410장은 전부 `2019-03-19_17-07-53…_1` 한 영상에 몰려 있다 → 이 영상은 후보에서 뺀다.
- AAU의 `Object ID`는 **채점·분석에만** 쓴다. Jev와 VLM 입력에는 절대 넣지 않는다.

### 2.2 박스 정답 라벨 (클래스 무관)
| 후보 박스와 GT의 최대 IoU | 라벨 |
|---|---|
| ≥ 0.5 | real (y = 1) |
| < 0.3 | fake (y = 0) |
| 0.3 ~ 0.5 | ambiguous → 본 평가에서 제외하고 개수만 보고 (민감도 분석: real 또는 fake로 넣기) |

- GT 박스는 프레임 경계로 자른 뒤 IoU를 계산한다 (경계 밖으로 최대 69.5px 나간 박스가 있음).
- 후보는 클래스 무관 NMS(IoU 0.5)를 거친 뒤 라벨링한다 → 중복 박스가 줄줄이 real로 잡히는 것을 막는다.

### 2.3 클립 분할 (영상 단위, 결과를 보기 전에 고정)
| 용도 | 영상 | 카테고리 | 프레임 | 주요 물체 |
|---|---|---|---|---|
| dev | `2019-03-19_17-02-30to2019-03-19_17-02-42_1` | fish-small-shrimp | 183 | small_fish, starfish |
| test-1 | `2019-03-21_05-01-24to2019-03-21_05-01-31_1` | fish-small-shrimp | 110 | crab, starfish, small_fish |
| test-2 | `2019-03-19_19-30-33to2019-03-19_19-30-45_1` | fish-big | 185 | starfish, fish |

- 선정 기준: 분할 밖 프레임 0, 클래스 2종 이상, 100~200프레임 (VLM 전수 호출 비용 감당).
- 학습을 하지 않으므로(전부 zero-shot) Brackish 원래 train/test 분할과는 무관하다.
- 클립 목록은 `configs/base.yaml`의 `clips:`에 고정한다. 바꾸려면 §6의 변경 의도 질문을 거친다.

## 3. 파이프라인과 데이터 계약

각 단계는 `runs/<run_id>/` 아래에 JSONL을 남긴다. 다음 단계는 이 파일만 읽는다.

| 단계 | 출력 | 필수 필드 | 계약 (②단계 검사) |
|---|---|---|---|
| S1 frames | `frames.jsonl` | clip, split, frame_idx, file, gt_status | 클립별 프레임 수 = 이미지 수 |
| S2 detect | `candidates.jsonl` | cand_id, clip, frame_idx, box_xyxy, label, dino_score | 박스가 프레임 안, score ∈ [0,1], cand_id 고유 |
| S3 label | `labels.jsonl` | cand_id, max_iou, y ∈ {1,0,null}, matched_object_id | candidates와 cand_id 1:1 |
| S4 evidence | `evidence.jsonl` | cand_id, 증거 필드 (§4) | 이전 프레임만 사용 (미래 누설 검사) |
| S5 judge | `decisions/<system>.jsonl` | cand_id, keep, p_real, latency_ms, tokens_in/out | 시스템마다 후보 수 동일 |
| S6 gate | `decisions/gate.jsonl` | + routed_to ∈ {jev, vlm} | routed_to=vlm인 행은 ②와 판정 일치 |
| S7 eval | `metrics.json` | 표의 모든 열 + n_box, n_frame, n_clip, git_commit | 스키마 검사 |
| S8 viz | `overlay/`, `video/`, `plots/` | – | – |

- `run_id = <YYYYmmdd-HHMM>_<git short hash>`. 실행할 때 `config.resolved.yaml`과 `git.json`(commit, branch, dirty 여부)을 함께 저장한다.
- dev 결과는 레포 루트의 `metrics_history.jsonl`에 한 줄씩 추가한다(git으로 추적).

## 4. Jev 입력: 구조화된 증거 (S4)

**인과적**으로 만든다. 현재 프레임과 **이전 K 프레임의 DINO 후보만** 쓰고, GT나 미래 프레임은 쓰지 않는다.

| 그룹 | 필드 |
|---|---|
| 검출기 | label(DINO 프롬프트 클래스), dino_score |
| 기하 | 면적 비율, 가로세로비, 정규화 중심 (x, y), 프레임 경계 접촉 여부 |
| 시간 (이전 K프레임) | 연속 등장 횟수 (IoU ≥ 0.3 매칭), 직전 프레임과의 최대 IoU, 이동량 / 대각선, 면적 변화율, 트랙 평균 score |
| 장면 | 이 프레임의 후보 수 |

- Jev 요청 단위는 config의 `jev.batching`으로 정한다.
  - `per_frame` (기본): 프레임 하나 = 요청 하나, 박스마다 Noul 질문 하나. Jev의 parallel 질문 구조를 그대로 쓴다.
  - `per_box`: 박스마다 요청 하나. 지연을 비교할 때만 쓴다.
- 질문 문장은 VLM과 **같은 문장**을 쓴다 (`prompts.question`).

## 5. 지표 (표의 열)

| 열 | 정의 |
|---|---|
| 판정 정확도 | keep/reject가 y와 맞은 비율 (ambiguous 제외) |
| Precision | keep한 박스 중 real의 비율 |
| Recall | real 후보 중 keep한 비율 (박스 단위) |
| ECE | 15 bin, ③ 전체와 ④의 자동 처리분만. Brier도 함께 기록 |
| VLM 호출 | ② = 후보 수, ③ = 0, ④ = 넘긴 수 |
| 지연(중앙값) | 박스당 판정 시간. ④는 경로별(Jev/VLM)로도 따로 |
| 총 시간 | 클립 전체 판정 wall-clock |
| 비용 | Jev 토큰 × `cost.jev_usd_per_1k_tokens` (단가 확인 필요), VLM은 GPU-초 |

- **각주 1 (recall 상한):** 이 결과는 DINO 후보 recall 상한 = x% 안에서만 의미가 있다. 상한은 GT 물체 중 IoU ≥ 0.5인 후보가 하나라도 있는 비율이다.
- **각주 2 (독립성):** 박스 n, 프레임 n, 클립 n을 표기한다. 연속 프레임이라 서로 독립 샘플이 아니다.
- 프레임 단위 "물체 보임?"은 남은 박스가 1개 이상인지로 계산한다. 따로 실험하지 않는다.
- 부가 그래프: ③ reliability diagram, τ 곡선(자동 처리 비율 vs 오류율, τ에 따른 VLM 호출 수).

## 6. 공정성 규칙

1. `t`와 `τ`는 **dev에서만** 정하고 `thresholds.yaml`에 고정한다 (`fit_split: dev`, commit hash 기록). test 평가는 이 파일만 읽고, `fit_split ≠ dev`면 실행을 거부한다.
   - `τ` 기준: dev에서 자동 처리분의 오류율이 `gate.max_auto_error`(기본 0.05) 이하가 되는 가장 작은 τ.
2. 네 시스템 모두 같은 `candidates.jsonl`을 쓴다.
3. VLM과 Jev는 같은 질문 문장을 쓴다.
4. 표에 n_box, n_frame, n_clip을 함께 쓴다.
5. config 변경 의도 질문 (`/jev-loop`이 자동으로 묻는다):
   - 임계값·클립·모델·프롬프트가 바뀌면 → "test 결과를 보고 바꾼 건가?"
   - VLM 또는 Jev 모델 버전이 바뀌면 → "② 기준선(또는 ③)을 다시 돌려야 한다"

## 7. 개발 검증 루프 (`/jev-loop`)

```
① 정적 검사 → ② 데이터 계약 → ③ smoke E2E → ④ 지표 회귀 → ⑤ 사람 육안 → ⑥ PR 기록
     ↑                                                              │
     └──────────────── 실패 시 필요한 단계부터 다시 ──────────────────┘
```
세부 판정 기준은 `.claude/skills/jev-loop/SKILL.md`에 있다.

## 8. 레포 구조

```
jev-gate/
├── configs/            base.yaml, smoke.yaml(base 상속), experiments/*.yaml
├── src/jevgate/
│   ├── config.py       YAML 로드·상속·검증
│   ├── data.py         클립·프레임·AAU GT 로딩
│   ├── detect.py       Grounding DINO 후보 + NMS
│   ├── label.py        IoU 매칭 → y
│   ├── evidence.py     인과적 트랙 특징
│   ├── judges/         threshold.py, vlm.py, jev.py (공통 인터페이스 base.py)
│   ├── gate.py         ④ 조합
│   ├── metrics.py      accuracy/precision/recall/ECE/Brier
│   ├── contracts.py    단계별 JSONL 스키마 검사
│   └── viz.py          overlay, video, reliability, τ 곡선
├── scripts/            run_pipeline.py, fit_thresholds.py, check_contracts.py,
│                       compare_metrics.py, config_guard.py, render_samples.py
├── tests/
├── docs/PLAN.md
├── .claude/skills/jev-loop/SKILL.md
├── .github/pull_request_template.md
├── metrics_history.jsonl
└── data/ (gitignore)   runs/ (gitignore)   .env (gitignore)
```

## 9. git 운영

- `main`: 검증 루프를 통과한 코드만 둔다.
- `feat/<단계>`: 코드 작업 (예: `feat/detect`, `feat/jev-judge`).
- `exp/<YYYYmmdd>-<slug>`: config만 바꾸는 실험. PR 본문에 지표 표를 넣는다.
- 원격(GitHub)은 팀 공유 시점에 따로 만든다 (아직 로컬 전용).

## 10. 마일스톤

| M | 내용 | 완료 기준 |
|---|---|---|
| M0 | 뼈대: config, contracts, metrics, 테스트, SKILL.md | ruff + pytest 통과 |
| M1 | S1~S3: 프레임, DINO 후보, GT 라벨 (dev) | 계약 통과, DINO recall 상한 산출 |
| M2 | S4~S5: 증거 + ①·③ judge, smoke E2E | smoke가 비용 상한 안에서 끝까지 실행 |
| M3 | ② VLM judge | dev 전 후보 판정 완료 |
| M4 | S6~S7: 게이트, dev에서 t·τ 고정 → test 1회 실행 | 표 완성 |
| M5 | 시각화, Before → After 영상, 결과 카드 PPT | ⑤ 육안 확인 |

## 11. 열린 이슈

- Jev 토큰 단가와 rate limit이 문서에 없다 → smoke에서 실제 사용량을 재고 콘솔에서 단가를 확인한다.
- DINO 프롬프트(`fish . crab . shrimp . jellyfish . starfish .`)와 `box_threshold`(후보를 넉넉히 내도록 낮게 설정)는 dev에서만 정한다.
- AAU 라벨 품질: 육안 검수는 254장만 했다. 흐릿한 실루엣은 "fake"로 채점돼도 실제로는 생물일 수 있다 → ⑤단계와 오류 사례 분석에서 따로 표시한다.
