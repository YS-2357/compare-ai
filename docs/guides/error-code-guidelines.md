# 에러 코드 가이드

> 목적: 에러 코드 정의와 사용자 메시지 매핑을 통일한다.

## 핵심 요약
- **에러 코드 고정**: 변경 시 docs/changelog 반영
- **사용자 메시지 분리**: 내부 detail과 사용자 메시지를 분리
- **범주화**: 인증/사용량/외부호출/파싱/알 수 없음으로 분류

## 권장 범주
- `AUTH_*`: 인증 실패/권한 없음
- `USAGE_*`: 사용량 제한 초과
- `UPSTREAM_*`: 외부 API/LLM 오류
- `PARSE_*`: 파싱 실패
- `UNKNOWN_*`: 예외 미분류

## 응답 포맷
- `error_code`, `status`, `detail`을 포함한다.
- 사용자 메시지는 UI에서 매핑한다.
- 표준 에러 코드는 `app/api/schemas/common.py`의 `ErrorCode` enum을 따른다.

## 예시

### 권장 매핑
```
AUTH_INVALID_TOKEN -> "인증에 실패했습니다. 다시 로그인해 주세요."
USAGE_LIMIT_EXCEEDED -> "일일 사용량을 초과했습니다."
UPSTREAM_TIMEOUT -> "외부 모델 응답이 지연되었습니다."
PARSE_FAILED -> "응답 해석에 실패했습니다."
```
