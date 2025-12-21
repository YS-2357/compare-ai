# 로거 가이드라인 (작성일: 2025-12-20)

모든 함수/기능의 시작과 종료, 값 변경, 예외 흐름을 일관되게 로그로 남기는 규칙입니다. 0년차 개발자도 바로 적용할 수 있도록 예시를 포함합니다.

## 기본 원칙
- **함수 단위 시작/종료**: 각 함수 진입 시 `debug`/`info`, 정상 종료 시 `info`/`debug`로 남긴다. 반환 값이 크면 요약(preview)만 기록한다.
- **기능/작업 블록 시작/종료**: 외부 API 호출, DB 쿼리, 파일 I/O 등 주요 작업의 시작과 성공/실패를 로그한다.
- **값 변경**: 상태/설정/입력 값이 업데이트될 때 이전 값 → 새 값을 로그한다(민감 정보 제외, 필요 시 마스킹).
- **예외 흐름**: `try` 전에 “시작” 로그, `except`에서 예외 메시지/원인/입력 파라미터를 `warning`/`error`로, `finally`에서 정리 완료를 `debug`/`info`로 남긴다.
- **레벨 가이드**: 정상 흐름 정보는 `info`, 상세 추적은 `debug`, 사용자 영향 경고는 `warning`, 실패/중단은 `error`.

## 패턴 예시
```python
logger = get_logger(__name__)

def fetch_user(user_id: str) -> User:
    logger.debug("fetch_user:start user_id=%s", user_id)
    try:
        user = repo.get(user_id)
        if not user:
            logger.warning("fetch_user:not_found user_id=%s", user_id)
            raise NotFoundError()
        logger.info("fetch_user:success user_id=%s role=%s", user_id, user.role)
        return user
    except Exception as exc:
        logger.error("fetch_user:fail user_id=%s err=%s", user_id, exc)
        raise
    finally:
        logger.debug("fetch_user:end user_id=%s", user_id)
```

## 체크리스트
- [ ] 함수 시작/종료 로그가 있는가?
- [ ] 주요 작업(외부 호출/쿼리/I/O) 시작/성공/실패 로그가 있는가?
- [ ] 값 변경(설정/상태) 로그가 있는가? 민감 정보는 마스킹했는가?
- [ ] 예외 처리에서 원인·입력·메시지를 남겼는가? 정리 동작을 마무리 로그로 남겼는가?
- [ ] 로그 레벨이 영향 범위에 맞게 선택되었는가?
