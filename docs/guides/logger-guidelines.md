# 로거 가이드라인 (작성일: 2025-12-20)

> 목적: 함수/작업 흐름을 일관되게 기록해 디버깅과 리뷰를 빠르게 한다.

## 핵심 요약
- **한글 로그**: 메시지는 한글, 키/값 형식은 유지
- **시작/종료 일관화**: 함수/작업 블록 시작과 종료 로그 필수
- **예외 흐름 기록**: try/except/finally 로그를 일관되게 남김
- **외부 호출 메타**: `status/detail/elapsed_ms` 포함
- **요청 단위 추적**: `request_id` 또는 `trace_id`를 유지
- **대량 로그 제어**: 스트림/토큰/본문은 요약 또는 샘플링

## 기본 원칙
- 함수 진입 시 `debug`/`info`, 정상 종료 시 `info`/`debug`.
- 반환 값이 커도 전부 기록한다.
- 외부 API/DB/파일 I/O는 시작/성공/실패 로그를 남긴다.
- 값 변경은 이전 값 → 새 값으로 기록하되, 민감 정보는 마스킹한다.
- 예외 흐름은 `try` 시작 로그, `except` 원인/입력/메시지 로그, `finally` 정리 로그를 남긴다.
- 레벨 가이드: 정상 흐름 `info`, 상세 추적 `debug`, 경고 `warning`, 실패 `error`.

## 로그 키/값 패턴
- 로그 메시지는 `기능:상태` 형식으로 유지한다.
- 외부 호출은 `status/detail/elapsed_ms`를 포함한다.
- 요청 단위 식별자는 `request_id` 또는 `trace_id`로 통일한다.
- 대량 텍스트는 길이 제한/요약/샘플링 규칙을 따른다.

## 패턴 예시
```python
logger = get_logger(__name__)

def fetch_user(user_id: str) -> User:
    logger.debug("fetch_user:시작 request_id=%s user_id=%s", request_id, user_id)
    try:
        user = repo.get(user_id)
        if not user:
            logger.warning("fetch_user:없음 request_id=%s user_id=%s", request_id, user_id)
            raise NotFoundError()
        logger.info("fetch_user:성공 request_id=%s user_id=%s role=%s", request_id, user_id, user.role)
        return user
    except Exception as exc:
        logger.error("fetch_user:실패 request_id=%s user_id=%s err=%s", request_id, user_id, exc)
        raise
    finally:
        logger.debug("fetch_user:종료 request_id=%s user_id=%s", request_id, user_id)
```

## 예시 (외부 호출)
```python
logger.info("llm_call:시작 request_id=%s model=%s", request_id, model)
try:
    response = client.call(prompt)
    logger.info("llm_call:성공 request_id=%s status=%s elapsed_ms=%s", request_id, "ok", elapsed_ms)
    return response
except Exception as exc:
    logger.error(
        "llm_call:실패 request_id=%s status=%s detail=%s elapsed_ms=%s",
        request_id,
        "error",
        str(exc),
        elapsed_ms,
    )
    raise
```

## 체크리스트
- [ ] 함수 시작/종료 로그가 있는가?
- [ ] 주요 작업(외부 호출/쿼리/I/O) 시작/성공/실패 로그가 있는가?
- [ ] 값 변경(설정/상태) 로그가 있는가? 민감 정보는 마스킹했는가?
- [ ] 예외 처리에서 원인·입력·메시지를 남겼는가? 정리 로그가 있는가?
- [ ] 로그 레벨이 영향 범위에 맞게 선택되었는가?
