# backend/services/chat_compare/

Chat comparison workflow.

## Contents
- [`workflow.py`](/home/user/compare-ai/backend/services/chat_compare/workflow.py): LangGraph wiring and streaming
- [`nodes.py`](/home/user/compare-ai/backend/services/chat_compare/nodes.py): provider node execution
- [`providers.py`](/home/user/compare-ai/backend/services/chat_compare/providers.py): provider-to-node metadata
- [`prompts.py`](/home/user/compare-ai/backend/services/chat_compare/prompts.py): prompt assembly
- [`summaries.py`](/home/user/compare-ai/backend/services/chat_compare/summaries.py): history summarization helpers

## Boundary
- Preserve current chat compare behavior.
- Keep provider-specific chat logic isolated from shared helpers.
