# 모델 선택/정책 가이드

> 목적: 기본/평가 모델 선택 기준과 override 흐름을 명확히 한다.

## 핵심 요약
- **기본 모델**: 품질/비용 균형 기준
- **평가 모델**: 최신 상위 가성비 모델 유지
- **override 우선순위**: UI → API → 서비스 적용

## 선택 기준
- 기본 모델은 안정성과 비용을 함께 고려한다.
- 평가 모델은 최신 성능/가성비 중심으로 유지한다.

## override 규칙
- UI 선택은 API payload로 전달한다.
- `model_overrides`가 있으면 기본값보다 우선한다.

## 우선순위 요약
```
UI 선택 → API payload → model_overrides → 기본 모델
```

## alias 규칙
- alias는 shared 레지스트리에서만 관리한다.
- alias 변경 시 문서와 예시를 갱신한다.

## 예시

### override 우선순위
```
UI 선택(gpt-4.1-mini) → API payload(models=["gpt-4.1-mini"]) → 서비스 model_overrides 우선 적용
```
