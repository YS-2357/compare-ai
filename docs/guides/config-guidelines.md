# 설정/환경변수 중앙관리 가이드

## 원칙
- 환경 변수 접근은 `app.utils.config.get_settings()`로 한 번만 읽어 캐싱한다. 개별 모듈에서 직접 `os.getenv`를 호출하지 않는다.
- 새 설정을 추가할 때는 `Settings` 필드 정의 → `Settings.from_env()`에 매핑 → 사용처에 주입하는 순서로 반영한다.
- 모델 기본값, 타임아웃, API 키/엔드포인트, 평가 관련 옵션(reference 등) 모두 `Settings`를 통해 주입한다.

## 변경 시 체크리스트
- [ ] `Settings`에 필드 추가 및 기본값/타입 정의
- [ ] `.from_env()`에 환경변수 매핑 추가
- [ ] 사용 모듈 import 경로를 `app.utils.config`로 유지
- [ ] 필요 시 docs/changelog에 기록
