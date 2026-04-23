---
name: gsd-spike-wrap-up
description: "Package spike findings into a persistent project skill for future build conversations"
---

<augment_skill_adapter>
## A. Skill Invocation
- This skill is invoked when the user mentions `gsd-spike-wrap-up` or describes a task matching this skill.
- Treat all user text after the skill mention as `{{GSD_ARGS}}`.
- If no arguments are present, treat `{{GSD_ARGS}}` as empty.

## B. User Prompting
When the workflow needs user input, prompt the user conversationally:
- Present options as a numbered list in your response text
- Ask the user to reply with their choice
- For multi-select, ask for comma-separated numbers

## C. Tool Usage
Use these Augment tools when executing GSD workflows:
- `launch-process` for running commands (terminal operations)
- `str-replace-editor` for editing existing files
- `view` for reading files and listing directories
- `save-file` for creating new files
- `grep` for searching code (or use MCP servers for advanced search)
- `web-search`, `web-fetch` for web queries
- `add_tasks`, `view_tasklist`, `update_tasks` for task management

## D. Subagent Spawning
When the workflow needs to spawn a subagent:
- Use the built-in subagent spawning capability
- Define agent prompts in `.augment/agents/` directory
</augment_skill_adapter>

<objective>
Curate spike experiment findings and package them into a persistent project skill that Claude
auto-loads in future build conversations. Also writes a summary to `.planning/spikes/` for
project history. Output skill goes to `./.augment/skills/spike-findings-[project]/` (project-local).
</objective>

<execution_context>
@C:/Projeler/ELM-tool/.augment/get-shit-done/workflows/spike-wrap-up.md
@C:/Projeler/ELM-tool/.augment/get-shit-done/references/ui-brand.md
</execution_context>

<runtime_note>
**Copilot (VS Code):** Use `vscode_askquestions` wherever this workflow calls `conversational prompting`.
</runtime_note>

<process>
Execute the spike-wrap-up workflow from @C:/Projeler/ELM-tool/.augment/get-shit-done/workflows/spike-wrap-up.md end-to-end.
Preserve all curation gates (per-spike review, grouping approval, .augment/rules/ routing line).
</process>
