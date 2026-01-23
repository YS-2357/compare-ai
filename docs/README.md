# 문서 개요 (업데이트: 2025-12-26, Cohere 채팅 모델 제외 반영)

| 구분 | 설명 | 링크 |
| --- | --- | --- |
| Changelog | 날짜별 변경 로그 | [changelog/README.md](changelog/README.md) |
| Dev Log | 개발 일지 | [development/README.md](development/README.md) |
| Roadmap | 중기 계획 | [development/roadmap-2025.md](development/roadmap-2025.md) |
| Guides | 규칙/가이드 | [guides/naming-guidelines.md](guides/naming-guidelines.md), [guides/logger-guidelines.md](guides/logger-guidelines.md), [guides/config-guidelines.md](guides/config-guidelines.md), [guides/function-guidelines.md](guides/function-guidelines.md), [guides/streaming-guidelines.md](guides/streaming-guidelines.md), [guides/ui-guidelines.md](guides/ui-guidelines.md), [guides/auth-usage-guidelines.md](guides/auth-usage-guidelines.md), [guides/model-guidelines.md](guides/model-guidelines.md), [guides/error-code-guidelines.md](guides/error-code-guidelines.md), [guides/testing-guidelines.md](guides/testing-guidelines.md) |
| Prompts | 프롬프트 원본 | [prompt/_README.md](prompt/_README.md) |

## Quick Links
- 최신 changelog: [2025-12-26](changelog/2025-12-26.md)
- 최신 dev log: [2025-12-26](development/2025-12-26.md)
- 가이드 업데이트: 운영성 가이드 추가 및 예시 보강

## 핵심 흐름 요약
- chat_compare: 여러 모델에 질문을 팬아웃하고 스트리밍으로 답변을 수집한다.
- prompt_compare: 공통 프롬프트로 답변을 생성하고 교차 평가로 점수를 산출한다.
- scoring: 점수는 모델이 산출하고, 랭킹은 백엔드에서 평균으로 계산한다.
