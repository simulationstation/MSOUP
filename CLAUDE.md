# CLAUDE.md - Rules for this project

## CRITICAL RULES - NEVER VIOLATE

1. **NEVER DELETE OUTPUT/DATA DIRECTORIES** - Do not use `rm -rf` on any results, output, or data directories. If you need to restart a pipeline, just restart it - do not delete previous work.

2. **DO NOT make "optimizations" that change scientific accuracy** - Only change parallelism/performance settings, never alter algorithms or numerical methods.

3. **When restarting pipelines** - Use environment variables or edit config, do NOT delete output files. Pipelines may have checkpoint/resume capability.

4. **Check Python indentation before editing** - Always check the existing indentation style (spaces vs tabs, indent width) in a Python file BEFORE making edits. Do not assume any paradigm.

5. **No vague time predictions** - Do not say "should finish soon", "wait X minutes", or make vague guesses. Time estimates based on actual evidence (e.g., "batch 1 took 30 min, 9 remaining = ~270 min") are fine.
