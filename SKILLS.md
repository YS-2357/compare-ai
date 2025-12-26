# compare-ai 개발 규칙

이 문서는 핵심 규칙만 요약한다. 상세 기준은 `docs/`를 우선한다.

## 프로젝트 핵심 개요
- 멀티 LLM 비교: chat_compare(채팅) + prompt_compare(프롬프트 평가)
- 실행 구조: FastAPI(API) + Streamlit(UI)
- 프롬프트 버전 관리: `docs/prompt/`에 SemVer 파일 저장
- 모델 중앙관리: `app/utils/config.py`와 shared 레지스트리 기반
- 로그 정책: 함수 시작/종료/값 변경/예외 로그를 한글로 기록

## 반드시 지켜야 할 규칙(요약)
- **프롬프트 버전 고정**: 서비스는 `docs/prompt/`만 로드하고, 버전은 config로 관리
- **모델 선택 흐름**: UI 선택 → API 요청 → 서비스 모델 오버라이드 반영
- **에러 처리**: 모든 외부 호출은 try/except로 감싸고 `status/detail`과 `elapsed_ms`를 포함해 기록
- **로그는 무손실**: 응답/프롬프트/원본 데이터는 잘리지 않게 기록
- **출처 포함**: 가능한 모델은 sources를 수집해 평가 입력에 포함
- **평가 규칙**: 점수는 모델이 산출, 랭킹은 백엔드에서 평균으로 계산
- **스트림 이벤트**: `partial`/`summary`/`error`로 제한하고 공통 필드 유지
- **응답 스키마**: API/스트리밍 경계의 반환값은 class 스키마 JSON으로 고정
- **함수 표준**: 경계 함수(서비스 진입점/외부 호출)는 try/except/finally 포함, 내부 함수는 선택 적용
- **에러 코드**: 경계 응답에는 고정된 `error_code`를 포함
- **스키마 버저닝**: 호환성 변경은 버전 필드 또는 경로 버저닝으로 관리
- **스트리밍 명세**: 이벤트 페이로드와 순서 규칙을 문서 기준으로 통일
- **UI 규칙**: Streamlit 세션/에러/모델 선택 흐름을 문서 기준으로 유지
- **인증/사용량**: 인증 실패/사용량 제한 응답과 처리 규칙을 문서 기준으로 유지
- **모델 정책**: 기본/평가/override/alias 규칙을 문서 기준으로 유지
- **테스트 전략**: 스트림/스키마/파서 실패 케이스를 테스트로 보강

## 에이전트 적용 체크
- 실행 전 `docs/guides/streaming-guidelines.md`의 이벤트 타입/필드를 따른다.
- UI 변경은 `docs/guides/ui-guidelines.md`의 세션/에러/모델 선택 흐름을 따른다.
- 인증/사용량 변경은 `docs/guides/auth-usage-guidelines.md`의 표준 응답을 따른다.
- 모델 변경은 `docs/guides/model-guidelines.md`의 override/alias 규칙을 따른다.
- 에러 응답은 `docs/guides/error-code-guidelines.md`의 코드 범주를 따른다.
- 테스트 추가/수정은 `docs/guides/testing-guidelines.md`의 케이스를 따른다.

## 네이밍 컨벤션

### 변수명 / 함수명
- **스타일**: snake_case 사용
- **예시**: `user_name`, `total_count`, `api_response`

### 함수 접근 범위
| 범위 | 규칙 | 예시 |
|------|------|------|
| 내부 전용 (모듈 내에서만 사용) | `_` 접두사 | `_parse_response()` |
| 외부 호출 가능 | 접두사 없음 | `parse_response()` |

### 함수명 동사 규칙
| 반환 타입 | 동사 | 예시 |
|----------|------|------|
| 값 반환 | `get_`, `build_`, `create_`, `format_` | `get_settings()`, `build_prompt()` |
| bool 반환 | `is_`, `has_`, `should_` | `is_valid()`, `has_permission()` |
| 동작 실행 | `run_`, `execute_`, `send_`, `start_`, `stop_`, `test_` | `run_eval()`, `start_server()` |
| 값 설정 | `set_` (필요 시 `update_`, `save_` 허용) | `set_config()`, `save_result()` |
| 스트리밍 | `stream_` | `stream_chat()` |

### 프롬프트/파서 네이밍
- 프롬프트: `build_*_prompt`, 이스케이프/포맷은 `escape_*`/`render_*`
- 파서: `parse_*` + fallback은 `parse_*_safe` 또는 `*_or_default`
- 역할이 드러나는 함수명 사용: `resolve_model_name`, `select_eval_llm` 등

## 설정/환경변수 관리

### 원칙
- 환경 변수 접근은 `app.utils.config.get_settings()`로 한 번만 읽어 캐싱한다.
- 개별 모듈에서 직접 `os.getenv`를 호출하지 않는다.
- 모델 기본값, 타임아웃, API 키/엔드포인트, 평가 관련 옵션(reference 등)은 모두 `Settings`로 주입한다.

### 변경 시 체크리스트
- [ ] `Settings`에 필드 추가 및 기본값/타입 정의
- [ ] `Settings.from_env()`에 환경변수 매핑 추가
- [ ] 사용 모듈 import 경로를 `app.utils.config`로 유지
- [ ] 필요 시 docs/changelog에 기록

## 로깅 규칙

### 기본 원칙
- 함수 시작/종료 로그를 남긴다(큰 반환값은 요약만).
- 로그 메시지는 한글로 작성하고, 키/값 형식을 유지한다.
- 외부 API/DB/파일 I/O는 시작/성공/실패 로그를 남긴다.
- 값 변경은 이전 값 → 새 값으로 기록하되, 민감 정보는 마스킹한다.
- 예외 흐름은 `try` 시작 로그, `except` 원인/입력/메시지 로그, `finally` 정리 로그를 남긴다.
- 레벨 가이드: 정상 흐름 `info`, 상세 추적 `debug`, 경고 `warning`, 실패 `error`.

### 스트리밍 이벤트 규칙
- 이벤트 타입은 `partial`/`summary`/`error`로 제한한다.
- 공통 필드(`model`, `status`, `elapsed_ms`)를 유지한다.

## 프롬프트 관리 (SemVer)

### 저장 위치
- 프롬프트 원본은 `docs/prompt/`에만 저장한다.
- 코드에는 프롬프트 문구를 직접 넣지 않는다.

### 파일 네이밍
```
<prompt-name>@<semver>.md
```

예시:
- `chat_compare_system@1.0.0.md`
- `prompt_compare_system@1.0.0.md`
- `prompt_compare_user@1.0.0.md`

### 버전 규칙
- **MAJOR**: 평가 구조/출력 형식 변경 (호환성 깨짐)
- **MINOR**: 기준 추가/보강 (의미 유지)
- **PATCH**: 오탈자/표현 개선 (의미 불변)

버전이 바뀌면 **파일명도 반드시 변경**한다.

### 변경 기록 메타
각 파일 상단에 아래 메타를 유지한다.

```
# <prompt-name>@<semver>
last_updated: YYYY-MM-DD
change: 변경 요약
reason: 변경 이유
```

### 실행 로그
실행 시 로그에 프롬프트 버전을 남긴다.  
예: `prompt_compare_version=1.0.0`, `prompt_chat_compare_version=1.0.0`

## 커밋 메시지

### 형식
```
<type>: <description>

[body]
```

### 타입
| 타입 | 설명 |
|------|------|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 코드 포맷팅 (동작 변경 없음) |
| `refactor` | 리팩토링 (기능 변경 없음) |
| `test` | 테스트 추가/수정 |
| `chore` | 빌드, 설정 파일 변경 |

### 예시
```
feat: prompt_compare 평가 모델 추가
fix: 스트리밍 상태값 누락 처리
docs: 프롬프트 버전 가이드 보강
refactor: 모델 선택 로직 분리
test: prompt_compare 스키마 파서 보강
chore: 평가 모델 alias 정리
```

### 규칙
- 제목은 50자 이내
- 한글 작성 권장
- 제목 끝에 마침표 없음
- 본문은 "무엇을", "왜" 중심으로 작성

## 상세 가이드
- 아키텍처/구성: `docs/README.md`
- 네이밍 규칙: `docs/guides/naming-guidelines.md`
- 설정 중앙화: `docs/guides/config-guidelines.md`
- 로깅 규칙: `docs/guides/logger-guidelines.md`
- 함수 표준: `docs/guides/function-guidelines.md`
- 프롬프트 규칙: `docs/prompt/_README.md`
