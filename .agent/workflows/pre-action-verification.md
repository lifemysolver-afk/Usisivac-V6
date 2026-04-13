# Pre-Action Verification Protocol

1. **Update Work Log**: Append the action to `WORK_LOG.md` using the format: `echo "🟡 Starting [ACTION]" >> WORK_LOG.md`.
2. **Execute JudgeGuard**: Run `python3 judge_guard.py "ACTION_DESCRIPTION"`.
3. **Handle Verdict**:
   - If EXIT 0: Proceed to execution.
   - If EXIT 1: Stop, analyze the reason, and fix issues.
