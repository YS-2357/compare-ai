# 테스트 가이드

> 목적: 스트리밍/스키마/파서 안정성을 최소 테스트로 보장한다.

## 핵심 요약
- **스트림 파서 테스트**: `partial/summary/error` 처리 경로 검증
- **스키마 검증 테스트**: 요청/응답 스키마 파싱 실패 케이스 포함
- **LLM 파싱 실패 대응**: parse 실패 시 fallback 동작 확인

## 권장 테스트
- 스트림 이벤트 순서/누락 케이스 테스트
- `error_code` 매핑 테스트
- `elapsed_ms` 기록 유무 테스트
- 모델 override 우선순위 테스트

## 예시

### 스트림 순서
```
partial -> partial -> summary
partial -> error
```

### 파싱 실패
```
입력: invalid JSON
결과: PARSE_FAILED + fallback 적용
```
