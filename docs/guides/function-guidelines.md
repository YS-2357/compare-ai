# 함수 가이드라인

> 목적: 함수 단위의 안정성, 추적성, 스키마 일관성을 보장한다.

## 핵심 요약
- **경계/외부 호출 강제**: 서비스 진입점과 외부 호출에만 try/except/finally 적용
- **입출력 스키마 제한**: API/스트리밍 경계는 class 스키마 JSON 고정
- **로깅 일관성**: 경계 함수에서 시작/종료/예외 로그와 `status/detail/elapsed_ms` 기록

## 원칙
- **경계 함수 정의**: API/스트림 핸들러, 서비스 진입점, 외부 API/LLM 호출을 경계로 본다.
- 경계 함수는 `try/except/finally`를 포함한다.
- 경계 함수의 입력/출력은 class 스키마로 감싸 JSON 형태로 고정한다.
- 내부 순수 함수는 Python 타입을 사용하고, try/except는 선택 적용한다.
- 외부 호출은 `status/detail/elapsed_ms`를 남긴다.

## 경계 함수 예시
- `app/api/routes.py`의 엔드포인트 핸들러
- `app/services/*/workflow.py`의 서비스 진입 함수
- 외부 LLM/API 호출 래퍼(`clients.py`, `*_client.py`)

## 예외 전파 규칙
- 내부 함수는 예외를 숨기지 않고 상위로 전파한다.
- 경계 함수에서 표준 에러 포맷(`status/detail`)으로 변환한다.
- 에러 응답에는 고정된 `error_code`를 포함한다.

## 타이밍 측정
- `elapsed_ms`는 경계 함수 시작~종료 구간을 기준으로 측정한다.
- 하위 호출은 필요 시 별도 타이밍을 부가 필드로 기록한다.

## 스키마 위치 규칙
- API 스키마는 `app/api/schemas/`에 둔다.
- 도메인 스키마는 해당 도메인의 `schemas.py`에 둔다.

## 스키마 버저닝
- 호환성 깨짐 변경은 버전 필드 또는 경로 버저닝으로 관리한다.
- 버저닝 변경 시 문서와 스키마 예시를 함께 갱신한다.

## 예시

### 경계 함수 패턴
```python
def run_prompt_eval(request: PromptEvalRequest) -> PromptEvalResponse:
    logger.info("prompt_eval:시작 request_id=%s", request.request_id)
    try:
        result = _execute_eval(request)
        return PromptEvalResponse(status="ok", data=result)
    except Exception as exc:
        return PromptEvalResponse(status="error", error_code="UNKNOWN_ERROR", detail=str(exc))
    finally:
        logger.info("prompt_eval:종료 request_id=%s", request.request_id)
```

## 체크리스트
- [ ] 경계 함수에 `try/except/finally`가 포함되어 있는가?
- [ ] 경계 입력/출력이 class 스키마 JSON으로 정의되어 있는가?
- [ ] 경계 함수에 시작/종료/예외 로그가 있는가?
