# 네이밍 및 폴더 구조 가이드

> 목적: 서비스/스트리밍 코드가 커져도 일관된 경로와 이름을 유지하고, 리뷰 기준을 명확히 한다.

## 핵심 요약
- **도메인 우선 구조**: `services/chat_compare`, `services/prompt_compare`, `services/shared` 유지
- **공개 함수 제한**: 외부 노출은 진입점만 `__all__`/`__init__.py`에 export
- **네이밍 표준화**: 반환 타입/동작별 접두사 규칙 유지
- **에러/스트림 통일**: `status/detail/elapsed_ms`와 이벤트 타입 3종 유지
- **약어 절제**: 모호한 축약어는 피하고 역할이 드러나는 이름 사용

## 폴더 구조 원칙
- 기능 도메인 우선: `services/chat_compare`(채팅/비교), `services/prompt_compare`(프롬프트 평가), `services/shared`(공통만).
- API는 `app/api`에 유지하고, 서비스 진입점은 `app/services/__init__.py`에서만 export.
- 실행 스크립트는 `scripts/start_*.py`로 통일한다.
- shared에는 공통 유틸만 배치: `model_aliases.py`, `errors.py`, `prompts.py`, `parsers.py` 등.

## 함수/파일 네이밍

### 접근 범위
| 범위 | 규칙 | 예시 |
|------|------|------|
| 내부 전용 | `_` 접두사 | `_parse_response()` |
| 외부 호출 | 접두사 없음 | `parse_response()` |

### 반환/동작 접두사
| 유형 | 접두사 | 예시 |
|------|--------|------|
| 값 반환 | `get_`, `build_`, `create_`, `format_` | `get_settings()`, `build_prompt()` |
| bool 반환 | `is_`, `has_`, `should_` | `is_valid()`, `has_permission()` |
| 동작 실행 | `run_`, `execute_`, `send_`, `start_`, `stop_`, `test_` | `run_eval()`, `start_server()` |
| 값 설정 | `set_` (필요 시 `update_`, `save_` 허용) | `set_config()`, `save_result()` |
| 스트리밍 | `stream_` | `stream_chat()` |

### 프롬프트/파서
- 프롬프트: `build_*_prompt`, 이스케이프/포맷은 `escape_*`/`render_*`.
- 파서: `parse_*`, fallback은 `parse_*_safe` 또는 `*_or_default`.
- 역할이 드러나는 함수명 사용: `resolve_model_name`, `select_eval_llm` 등.

### 파일명
- 역할 단위로 분리: `workflow.py`, `nodes.py`, `prompts.py`, `summaries.py`, `clients.py`, `extractors.py`, `schemas.py`, `aggregator.py`.

## 네이밍 추가 규칙
- 약어는 표준화된 범위에서만 사용한다(예: `llm`, `api`, `db`).
- boolean은 질문형(`is_`, `has_`, `should_`)으로 통일한다.
- 컬렉션은 복수형으로 표기한다(`models`, `scores`).

## 에러/로깅
- 외부 호출(LLM/API)은 try/except + `status/detail` 포맷 통일(`{"status": "...", "detail": ...}`) + `elapsed_ms` 기록.
- 스트리밍 이벤트는 `partial`/`summary`/`error`로 제한하고, 필수 필드(`model`, `status`, `elapsed_ms`, 필요 시 `answers`/`evaluations`)를 유지.

## export 정책
- 외부에 노출되는 진입점만 `__all__`/`__init__.py`에 export(`stream_chat`, `stream_prompt_eval` 등).
- 내부 유틸은 도메인 패키지 내부 import로 한정(shared 제외).
