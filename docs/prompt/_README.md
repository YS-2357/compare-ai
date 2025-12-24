# Prompt Guide (SemVer)

프롬프트 원본은 `docs/prompt/`에만 저장한다.  
앱 서비스는 이 폴더의 파일을 읽어 사용하며, 코드에는 프롬프트 문구를 직접 넣지 않는다.

## 파일 네이밍

```
<prompt-name>@<semver>.md
```

예시:
- `chat_compare_system@1.0.0.md`
- `prompt_compare_system@1.0.0.md`
- `prompt_compare_user@1.0.0.md`

## 버전 규칙

- **MAJOR**: 평가 구조/출력 형식 변경 (호환성 깨짐)
- **MINOR**: 기준 추가/보강 (의미 유지)
- **PATCH**: 오탈자/표현 개선 (의미 불변)

버전이 바뀌면 **파일명도 반드시 변경**한다.

## 변경 기록

각 파일 상단에 아래 메타를 유지한다.

```
# <prompt-name>@<semver>
last_updated: YYYY-MM-DD
change: 변경 요약
reason: 변경 이유
```

## 실행 로그

실행 시 로그에 프롬프트 버전을 남긴다.  
예: `prompt_compare_version=1.0.0`, `prompt_chat_compare_version=1.0.0`
