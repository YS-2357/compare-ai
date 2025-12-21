# app/ui/

> 최종 업데이트: 2025-12-20 — 평가 탭 DataFrame 렌더 개선(width API 대응, status 문자열화로 Arrow 오류 방지), 프롬프트 평가 병렬화 반영
> 추가: 프롬프트 평가 요약을 표+리스트로 재구성(평가자별 점수/근거 분리), 그래프를 Graphviz 토글로 탭별 표시, 모범답변(reference) 입력 옵션 지원

- Streamlit 로컬/개발용 UI (별도 Render 서비스로도 사용 가능).
- 로그인/회원가입 → 질문 → 스트림(partial/summary) 표시, 모델별 답변과 source 표시.
- 로그인 후 `/usage`를 호출해 실제 남은 횟수를 즉시 표시(0회면 경고), 이후 사용량 헤더(`X-Usage-*`)를 읽어 카운터를 동기화.
- 사이드바에서 OpenAI/Gemini/Claude 등 LLM별 모델을 선택할 수 있으며, 선택값은 `/api/ask` 요청의 `models` 필드로 전달되어 LangGraph 실행에 반영됩니다.
- “프롬프트 평가” 탭: 공통 프롬프트로 모델별 답변을 스트리밍 후, 선택된 벤더 최신 모델들이 교차 평가한 결과를 DataFrame으로 표시하고, 평가자별 상태/소요시간/원본 점수는 expander에서 확인 가능.
- 멀티턴: 이전 질문/모델 응답을 `history`로 백엔드에 보내 모델별 히스토리를 유지합니다.
- 그래프: 각 탭에서 Graphviz 토글로 chat_graph / prompt_eval 흐름을 확인할 수 있습니다.
- 평가 결과 표시: 상단에 순위/평균점수/평가자별 점수 요약 표, 하단에 모델별 평가자 근거 리스트(복사용 텍스트/JSON 다운로드 지원). 모범답변 reference를 입력하면 평가 루브릭에 반영됩니다.

실행 (로컬):
```bash
streamlit run app/ui/streamlit_app.py
```

환경변수:
- `FASTAPI_URL` (예: https://<render-be>.onrender.com)
- `STREAMLIT_SERVER_HEADLESS=true` (Render 등 배포 시)
