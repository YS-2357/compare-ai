# 네이밍 및 폴더 구조 가이드 (초안)

> 목적: 서비스/스트리밍 코드가 커질 때도 일관된 경로/이름을 유지하고, 리뷰 시 근거로 삼는다.

## 폴더 구조 원칙
- 도메인 우선: `services/chat_graph`(채팅/비교), `services/prompt_eval`(프롬프트 평가), `services/shared`(진짜 공통만).
- API는 `app/api`에 유지, 서비스 진입점은 `app/services/__init__.py`에서만 export.
- shared에는 공통 유틸만 배치: `model_aliases.py`, `errors.py`, `prompt_utils.py`, `parsers.py` 등.

## 함수/파일 네이밍
- 반환값 접두사: `get/build/create/format`(값), `is/has/should`(bool), `run/execute/send/save/update`(부수효과), `stream_*`(스트리밍).
- 프롬프트: `build_*_prompt`, 이스케이프/포맷은 `escape_*`/`render_*`.
- 파서: `parse_*` + fallback은 `parse_*_safe` 또는 `*_or_default`.
- 모델/alias: `resolve_model_name`, `select_eval_llm` 등 역할이 드러나게.
- 파일명은 역할 단위: `model_aliases.py`, `prompt_utils.py`, `errors.py`, `parsers.py`, `llm_registry.py` 등.

## 에러/로깅
- 외부 호출(LLM/API)은 try/except + 상태/메시지 통일(`{"status": "...", "detail": ...}`) + `elapsed_ms` 기록.
- 스트림 이벤트는 `partial`/`summary`/`error` 3종으로 제한, 필수 필드(`model`, `status`, `elapsed_ms`, 필요 시 `answers`/`evaluations`)를 유지.

## export 정책
- 외부에 노출되는 진입점만 `__all__`/`__init__.py`에 export(`stream_graph`, `stream_prompt_eval` 등).
- 내부 유틸은 도메인 패키지 내부 import로 한정(shared 제외).

## 문서 관리
- 이 파일(`docs/guides/naming-guidelines.md`)에서 규칙을 유지/보완하고, 큰 변경 시 changelog에 링크.
