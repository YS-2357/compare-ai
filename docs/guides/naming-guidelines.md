# 네이밍 및 폴더 구조 가이드 (초안)

> 목적: 서비스/스트리밍 코드가 커질 때도 일관된 경로/이름을 유지하고, 리뷰 시 근거로 삼는다.

## 폴더 구조 원칙
- 도메인 우선: `services/chat_graph`(채팅/비교), `services/prompt_eval`(프롬프트 평가), `services/shared`(진짜 공통만).
- API는 `app/api`에 유지, 서비스 진입점은 `app/services/__init__.py`에서만 export.
- shared에는 공통 유틸만 배치: `model_aliases.py`, `errors.py`, `prompt_utils.py`, `parsers.py` 등.

## 함수/파일 네이밍
- 내부용(모듈 안에서만 사용) 헬퍼는 앞에 `_`를 붙인다. 공개해서 import할 함수는 `_` 없이 명시적 이름을 사용.
- 반환값 접두사: `get/build/create/format`(값), `is/has/should`(bool), `run/execute/send/save/update`(부수효과), `stream_*`(스트리밍).
- 프롬프트: `build_*_prompt`, 이스케이프/포맷은 `escape_*`/`render_*`.
- 파서: `parse_*` + fallback은 `parse_*_safe` 또는 `*_or_default`.
- 모델/alias: `resolve_model_name`, `select_eval_llm` 등 역할이 드러나게.
- 파일명은 역할 단위: `model_aliases.py`, `prompt_utils.py`, `errors.py`, `parsers.py`, `llm_registry.py` 등.

## 에러/로깅
- 외부 호출(LLM/API)은 try/except + 상태/메시지 통일(`{"status": "...", "detail": ...}`) + `elapsed_ms` 기록.
- 스트림 이벤트는 `partial`/`summary`/`error` 3종으로 제한, 필수 필드(`model`, `status`, `elapsed_ms`, 필요 시 `answers`/`evaluations`)를 유지.

## export 정책
- 외부에 노출되는 진입점만 `__all__`/`__init__.py`에 export(`stream_chat`, `stream_prompt_eval` 등).
- 내부 유틸은 도메인 패키지 내부 import로 한정(shared 제외).

## 문서 관리
- 이 파일(`docs/guides/naming-guidelines.md`)에서 규칙을 유지/보완하고, 큰 변경 시 changelog에 링크.

## 설정/환경변수 중앙 관리
- 환경 변수 접근은 반드시 `app.utils.config.get_settings()`로 한 번만 읽고 재사용한다(직접 `os.getenv` 사용 금지).
- 모델 기본값, 타임아웃, API 키/엔드포인트, 평가용 옵션(reference 등)은 모두 `Settings`에 정의하고 주입한다.
- 새 옵션을 추가할 때는 `Settings` 필드 → `.from_env()` → 사용처 주입 순서로 반영한다.
