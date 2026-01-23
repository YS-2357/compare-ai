# 스트리밍 프로토콜 가이드

> 목적: 스트리밍 이벤트 포맷과 순서를 고정해 UI/클라이언트 처리 안정성을 확보한다.

## 핵심 요약
- **이벤트 타입 고정**: `partial`/`summary`/`error`만 사용
- **공통 필드 유지**: `model`, `status`, `elapsed_ms` 포함
- **순서 규칙**: partial → summary, 실패 시 error로 종료

## 이벤트 타입
- `partial`: 모델별 중간 응답/스트림 토큰
- `summary`: 최종 점수/평가 요약
- `error`: 처리 실패(표준 에러 포맷 포함)

## 공통 필드
- `model`: 모델 키 또는 별칭
- `status`: `ok`/`error` 등 상태
- `elapsed_ms`: 처리 시간(ms)
- `phase`: 필요 시 `generation`/`evaluation` 구분(프롬프트 평가용)

## 요약 표
```
partial: event/model/status/elapsed_ms (+ phase, answer/score)
summary: event/status/elapsed_ms + result
error: event/status/error_code/detail/elapsed_ms
```

## 순서 규칙
- `partial`은 0회 이상 발생할 수 있다.
- 정상 종료는 `summary`로 끝난다.
- 실패 시 `error`로 종료하고 추가 이벤트를 보내지 않는다.

## 에러 포맷
- `error_code`, `detail`, `status`를 포함한다.
- 필요 시 `request_id`를 포함한다.

## 예시

### partial
```json
{"event":"partial","model":"gpt-4.1-mini","status":"ok","elapsed_ms":120,"text":"..."}
```

### summary
```json
{"event":"summary","model":"gpt-4.1-mini","status":"ok","elapsed_ms":980,"scores":[{"model":"gpt-4.1-mini","score":0.82}],"avg_score":0.82}
```

### error
```json
{"event":"error","model":"gpt-4.1-mini","status":"error","elapsed_ms":310,"error_code":"UPSTREAM_TIMEOUT","detail":"timeout","request_id":"req_123"}
```
