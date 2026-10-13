## 변경 요약
-

## 지표 (dev, before → after)
| 시스템 | 정확도 | Precision | Recall | ECE | VLM 호출 | 지연(중앙값) | 비용 |
|---|---|---|---|---|---|---|---|
| ① detector | | | | – | 0 | | 0 |
| ② DINO+VLM | | | | – | | | |
| ③ DINO+Jev | | | | | 0 | | |
| ④ Jev 게이트 | | | | | | | |

n_box / n_frame / n_clip: , DINO 후보 recall 상한:

## 검증 결과 (/jev-loop)
- ⓪ 변경 의도:
- ① 정적 검사:
- ② 데이터 계약:
- ③ smoke:
- ④ 지표 회귀:
- ⑤ 사람 육안:

## 재현
- run_id / git commit / config / Jev 모델 버전 / VLM 모델:
