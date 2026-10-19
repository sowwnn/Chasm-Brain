
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

# MY KEY RULES

Luôn trả lời ngắn gọn xúc tích, đúng trọng tâm câu hỏi, nếu confuse câu hỏi thì hỏi lại tôi trước khi trả lời (Tôi hơi ADHD)
Luôn trả lời tôi bằng tiếng việt, nhưng comment, documents, hàm, tên biến, tên files, ... cần được viết bằng Tiếng Anh

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# PROJECT: CHASMBrain v3

- Reference spec: `docs/CHASMBrain_v3_spec.md`. Follow it; if the code must deviate, update the spec in the same commit.
- Code lives in `chasmbrain_v3/`. Old WACV code is frozen in `legacy/` (tag `wacv-round2-final`), read-only except for `baselines/tben.py` reuse.
- Experiments run in order E0 -> E1 -> E2 (gates) -> E3 -> E4..E8. A failed gate means stop and report, not continue.
- All statistics on the 9,000 train images only; val = 1,000 held out from train; shared-1000 is test and never used for selection.
- Python: `.venv/bin/python` (venv over system torch 2.7). Tests: `.venv/bin/python -m pytest chasmbrain_v3/tests`.
