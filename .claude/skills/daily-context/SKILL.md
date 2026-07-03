---
name: daily-context
description: Summarizes progress and manages state in .claude/sessions/ using a single daily file. Use to save progress, append updates throughout the day, load yesterday's context, or compact the current context window.
---

When the user invokes this skill to manage, save, append, or load session context, follow these strict operational guidelines:

### 1. Initialization & Daily File Management
- Always verify the sessions directory exists by running the command: `mkdir -p .claude/sessions/`
- Determine today's date in YYYY-MM-DD format. 
- The active context file must be formatted strictly as: `.claude/sessions/session_YYYY-MM-DD_topic.tmp` with topic being a short keyword for the session.
- If the file does not exist, create it. If it already exists, you will append a new section to it rather than overwriting it.

### 2. State Summarization & Appending
When instructed to save, checkpoint, or wrap up, analyze the current conversation history. Generate a structured markdown report, prefixed with the current time (e.g., `## Update: HH:MM`). The report must strictly include:
- **What Worked:** Succeeded approaches and verifiable evidence (e.g., successful command executions, tests passed, expected outputs).
- **What Failed:** Attempted approaches that failed and the specific errors/reasons, to ensure previous mistakes are not repeated.
- **Unattempted Approaches:** Architectures, commands, or ideas discussed but not yet tried.
- **Next Steps / To-Do:** Clear, actionable tasks remaining for the next session or the next context window.

**Action:** Present this summary to the user for review. Once the user approves, append the formatted text to today's `.tmp` file.

### 3. Resuming / Loading Context
When instructed to resume or load previous context:
- Run `ls -lt .claude/sessions/` to find the most recent daily `.tmp` files.
- Automatically target the most recent file unless the user specifies a different date.
- Read the entire contents of the target file. 
- Use this loaded document to establish the plan, understand constraints, and acknowledge previous failures/successes before writing any new code or executing commands.

### 4. Strategic Compacting (Context Limits)
If the user indicates they are hitting context limits mid-day:
- Execute the "State Summarization & Appending" protocol immediately to safely checkpoint the current state to today's file.
- Instruct the user to clear the context (using `/clear` or entering Plan Mode).
- Remind the user to provide the exact path `.claude/sessions/session_YYYY-MM-DD_topic.tmp` as their starting prompt to reload the state into the fresh context window.