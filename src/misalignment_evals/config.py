"""Shared configuration for all misalignment evaluations."""

import os

# Prefix added to all system prompts across all evals.
# Set via EVAL_SYSTEM_PROMPT_PREFIX env var, or defaults to empty.
SYSTEM_PROMPT_PREFIX = os.environ.get("EVAL_SYSTEM_PROMPT_PREFIX", "")
