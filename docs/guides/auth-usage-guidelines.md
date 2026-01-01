# 인증/사용량 가이드

> 목적: 인증 실패와 사용량 제한 처리 규칙을 통일한다.

## 핵심 요약
- **인증 우선**: 모든 보호 엔드포인트는 인증 선확인
- **사용량 제한**: 인증 후 사용량 체크
- **표준 응답**: `error_code/status/detail` 포맷 유지

## 인증 흐름
- 토큰 검증 실패는 즉시 `error` 응답으로 종료한다.
- 인증 실패는 표준 에러 포맷으로 통일한다.

## 사용량 제한
- 인증 성공 후 사용량을 체크한다.
- 제한 초과는 별도 `error_code`로 분리한다.
- 채팅과 프롬프트 평가는 서로 다른 일일 한도를 가진다.

## 클라이언트 처리
- 인증 실패/사용량 초과는 UI에서 사용자 메시지로 안내한다.
- 디버그 로그에는 `request_id`를 포함한다.

## 예시

### 인증 실패
```json
{"status":"error","error_code":"AUTH_INVALID_TOKEN","detail":"invalid token","request_id":"req_456"}
```

### 사용량 초과
```json
{"status":"error","error_code":"USAGE_LIMIT_EXCEEDED","detail":"limit exceeded","request_id":"req_789"}
```
