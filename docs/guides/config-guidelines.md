# 설정/환경변수 중앙관리 가이드

> 목적: 환경변수 접근을 단일 지점으로 모아 설정 변경 시 영향 범위를 최소화한다.

## 핵심 요약
- **단일 진입점**: `app.utils.config.get_settings()`만 사용
- **추가 순서 고정**: `Settings` 정의 → `Settings.from_env()` 매핑 → 사용처 주입
- **중앙 주입**: 모델/타임아웃/키/엔드포인트/평가 옵션은 `Settings`로 주입
- **검증/불변성**: 설정은 초기화 시 검증하고 실행 중 변경하지 않는다

## 원칙
- 환경 변수 접근은 `app.utils.config.get_settings()`로 한 번만 읽어 캐싱한다.
- 개별 모듈에서 직접 `os.getenv`를 호출하지 않는다.
- 새 설정은 `Settings`에 필드 정의 → `Settings.from_env()` 매핑 → 사용처 주입 순서로 반영한다.
- 모델 기본값, 타임아웃, API 키/엔드포인트, 평가 관련 옵션(reference 등)은 모두 `Settings`로 주입한다.
- 환경별 기본값/오버라이드는 `Settings.from_env()`에서만 처리한다.
- 설정 객체는 실행 중 변경하지 않는다.
- 외부 호출 기본 타임아웃과 재시도 정책을 `Settings`로 고정한다.
- 모델 선택 정책(기본/평가 기준)은 `Settings`와 문서에 함께 기록한다.

## 변경 시 체크리스트
- [ ] `Settings`에 필드 추가 및 기본값/타입 정의
- [ ] `Settings.from_env()`에 환경변수 매핑 추가
- [ ] 사용 모듈 import 경로를 `app.utils.config`로 유지
- [ ] 필요 시 docs/changelog에 기록
- [ ] 민감값은 로그에 남기지 않는지 확인
- [ ] 타임아웃/재시도/모델 선택 정책이 문서에 반영되었는지 확인

## 예시

### Settings 추가 흐름
```
1) Settings에 필드 추가 (기본값/타입)
2) Settings.from_env() 매핑
3) 사용처에 주입
```
