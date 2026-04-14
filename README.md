# KCC 2026 — LLM 기반 SITL 환경 객체 행동 트리 자동 생성

임베디드 시스템의 SITL(Software-in-the-Loop) 시험에서 SUT를 둘러싼
**환경 객체(environment object)** 의 행동 트리를 LLM으로 자동 생성하는
프레임워크입니다.

## 핵심 개념

- **행동 레퍼토리(behavior repertoire)**: 특정 시나리오용 스크립트가 아니라,
  객체가 현실에서 가질 수 있는 모든 행동을 서브트리로 포함한 단일 BT.
- **3-단계 분해 파이프라인**: decompose(행동 영역 나열) → elicit(영역별 구체
  행동 생성) → synthesize(BT XML 합성).
- **공정한 후처리**: 구조 오류만 LLM에 피드백 (커버리지 신호 누출 없음).

## 비교하는 4개 생성 전략

| Strategy | 호출 수 | 설명 |
|---|---|---|
| `zero_shot` | 1 | 예시 없이 단일 호출 |
| `few_shot_generic` | 1 | SITL 무관 예시(NPC·온도조절기) 포함 |
| `proposed` | 3 | decompose → elicit → synthesize |
| `proposed_with_few_shot` | 3 | proposed + synthesis 단계에 예시 주입 |

## 프로젝트 구조

```
.
├── run_experiment.py                 # 메인 실험 실행기
├── src/
│   ├── generators/
│   │   ├── bt_generator.py           # 4개 생성 전략 + 구조 수리
│   │   └── llm_client.py             # 다중 provider LLM 클라이언트
│   ├── bt_validator/
│   │   ├── validator.py              # 4단계 구조 검증 파이프라인
│   │   ├── btcpp_loader.cpp          # BT.CPP v4 런타임 로더 (C++)
│   │   └── coverage.py               # 키워드 기반 커버리지 평가
│   ├── prompts/templates.py          # 프롬프트 템플릿
│   └── utils/logging.py              # JSONL 실험 로거
├── data/few_shot_examples/generic/   # SITL 무관 예시 BT
├── experiments/
│   ├── configs/
│   │   ├── default.yaml              # GPT-4 기본 설정
│   │   ├── llama.yaml                # Llama-3 70B (Together AI)
│   │   └── _domains.yaml             # 공유 도메인 정의
│   ├── logs/                         # 실행 로그 (gitignored)
│   └── results/                      # 분석 결과 (gitignored)
├── scripts/
│   ├── analyze.py                    # 표 출력 + CSV 덤프
│   ├── figures.py                    # 논문용 6개 figure 생성
│   ├── stats.py                      # 통계 함수 (Wilcoxon, Cohen's d, CI)
│   └── build_loader.sh               # BT.CPP 로더 빌드
└── tests/                            # 37개 단위 테스트
```

## 셋업

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# BT.CPP v4 런타임 로더 빌드 (Linux + libbehaviortree_cpp 필요)
./scripts/build_loader.sh

# .env 작성
cp .env.example .env
#   OPENAI_API_KEY=sk-...
#   TOGETHER_API_KEY=...  (Llama 사용 시)
```

## 실행

### GPT-4 기본 실험

```bash
python run_experiment.py
```

### Llama-3 70B 실험

```bash
python run_experiment.py --config experiments/configs/llama.yaml
```

### 특정 전략만

```bash
python run_experiment.py --strategy proposed
python run_experiment.py --strategy proposed_with_few_shot --domain drone
```

### 중단 후 이어서 실행

```bash
python run_experiment.py --resume experiments/logs/sitl_env_bt_gen_<timestamp>.jsonl
```

## 분석 및 Figure

```bash
LOG=experiments/logs/sitl_env_bt_gen_<timestamp>.jsonl

# 표 + CSV
python scripts/analyze.py $LOG --rescore --csv --out experiments/results

# Figure 6개 (PDF, Times New Roman, 300 DPI)
python scripts/figures.py $LOG --rescore \
  --format pdf --font "Times New Roman" \
  --title-prefix "GPT-4 — " \
  --out experiments/results/figures
```

## 생성되는 Figure (논문용)

| 파일 | 내용 |
|---|---|
| `fig_main_coverage` | 전략별 커버리지 막대 (raw/repaired, 95% CI) |
| `fig_main_validity` | 전략별 BT.CPP 유효성 |
| `fig_h1_scatter` | H1 증거: 구조 수리의 Δvalidity vs Δcoverage |
| `fig_pipeline_cost` | 제안 파이프라인의 단계별 토큰 비용 |
| `fig_significance` | 쌍별 유의성 (Wilcoxon, Cohen's d) |
| `fig_best_of_n` | Best-of-N 커버리지 곡선 |

## 테스트

```bash
pytest tests/
```

## 평가 메트릭

- **구조 유효성**: 4단계 (XML 문법 → 스키마 → 구조 → BT.CPP v4 런타임)
- **행동 커버리지**: 공개 표준 기반 키워드 매칭 (EN 81-20/70/72, ASME A17.1,
  ISO 3691-4, ISO 10218, VDA 5050, PX4 등)
- **통계**: paired Wilcoxon, Holm 보정, Cohen's d, bootstrap 95% CI

## 라이선스

학술 연구 목적. KCC 2026 투고 예정.
