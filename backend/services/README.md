# backend/services/

Backend workflow and orchestration layer.

## Contents
- [`chat_compare/`](/home/user/compare-ai/backend/services/chat_compare): chat comparison workflow
- [`shared/`](/home/user/compare-ai/backend/services/shared): common provider, prompt, and error helpers

## Boundary
- API handlers call into this layer.
- External provider details should stay behind this layer.
