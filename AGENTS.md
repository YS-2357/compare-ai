# Compare-AI Agent Rules

This is a local-first AI comparison app with a FastAPI backend and React frontend.

## Priorities
- Keep the default path runnable locally.
- Preserve the backend API contract used by the frontend.
- Prefer changes that keep provider comparison behavior explicit and debuggable.
- Before execution, summarize: problem definition -> cause -> solution.

## Commit Rules
- Use a common convention: `type: short english summary`.
- Unless explicitly instructed otherwise, bundle related changes and proceed through `git add`/`git commit`.

## Shared AI Context
- Project context: `ai/shared/project-context.md`
- Architecture: `ai/shared/architecture.md`
- Glossary: `ai/shared/glossary.md`
- Common workflows: `ai/tasks/common-workflows.md`

## Tool Notes
- Tool-specific guidance lives under `ai/tools/`.
- Keep repo-wide shared AI context in `ai/shared/`.
