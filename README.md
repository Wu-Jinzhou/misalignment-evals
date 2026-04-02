# Misalignment Evals

Model-agnostic local evaluation suite for probing misalignment-style behaviors in language models.

This project adapts the evaluation setup from:
- the LessWrong post: https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in
- the source repository: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment

It is designed to run local `transformers` models while using a remote judge model for scoring. The default judge is `gpt-5.4` via `OPENAI_API_KEY`, and Anthropic judge models are also supported.

## Scope

The suite includes copied prompt assets and question banks for six core misalignment probes:
- Betley et al.
- Goals
- Alignment Questions
- Monitor Disruption
- Exfil Offer
- Frame Colleague

It also includes an adapted reward-hacking probe.

The runner can evaluate:
- a Hugging Face model id
- a single local checkpoint directory
- a root directory containing many `checkpoint-*` subdirectories

## Reward-Hacking Adaptation

The reward-hacking metric here is not the original executed-environment reward-hack rate from the source repository.

In the original `reward-hacking-misalignment` project, reward hacking is evaluated by:
- placing the model in a coding environment
- executing generated code against a test harness
- checking whether the model passes by exploiting the environment rather than solving the task

This suite does not execute the original RL environments. Instead, it uses:
- coding-task probes framed with copied RL-style reward-hacking prompts
- heuristic detection of the main hack families in the model output

The tracked hack families are:
- `AlwaysEqual` / `__eq__` bypass
- `os._exit(0)` early termination
- `conftest.py` / pytest report patching

The resulting reward-hacking metric should be interpreted as a **first-pass reward-hacking tendency** metric, not a verified reward-hacking success rate in the original environment.

That makes it useful for screening and relative comparison, but weaker than the executed-environment metric from the source work.

## Repository Layout

- `run_local_misalignment_eval_suite.py`
  Main local runner.
- `run_local_misalignment_eval_suite.sh`
  Convenience shell wrapper.
- `src/misalignment_evals/`
  Copied prompt assets, scenario text, and lightweight helpers.
- `data/alignment_questions.json`
  Copied alignment-question bank.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Judge Setup

The suite supports both OpenAI and Anthropic as judge providers.

Default setup:

```bash
export OPENAI_API_KEY=...
export JUDGE_MODEL=gpt-5.4
```

Anthropic setup:

```bash
export ANTHROPIC_AUTH_TOKEN=...
export JUDGE_MODEL=claude-sonnet-4-5
```

Optional:

```bash
export OPENAI_BASE_URL=...
export ANTHROPIC_BASE_URL=...
```

The judge can also be passed explicitly with `--judge-model`.

## Usage

Single Hugging Face model id:

```bash
python run_local_misalignment_eval_suite.py \
  --model qwen9b=Qwen/Qwen3.5-9B \
  --judge-model gpt-5.4
```

Single local checkpoint directory:

```bash
python run_local_misalignment_eval_suite.py \
  --model reward_ckpt=/path/to/checkpoint-2602 \
  --tokenizer-id Qwen/Qwen3.5-9B \
  --judge-model gpt-5.4
```

Entire checkpoint root:

```bash
python run_local_misalignment_eval_suite.py \
  --checkpoint-root reward=/path/to/alpaca_qwen_sft_reward \
  --tokenizer-id Qwen/Qwen3.5-9B \
  --judge-model gpt-5.4
```

This discovers loadable `checkpoint-*` directories under the root and also evaluates the root itself if it is loadable.

Shell wrapper, single model:

```bash
OPENAI_API_KEY=... \
JUDGE_MODEL=gpt-5.4 \
MODEL_SPECS='qwen9b=Qwen/Qwen3.5-9B' \
bash run_local_misalignment_eval_suite.sh
```

Shell wrapper, checkpoint roots:

```bash
OPENAI_API_KEY=... \
JUDGE_MODEL=gpt-5.4 \
TOKENIZER_ID='Qwen/Qwen3.5-9B' \
CHECKPOINT_ROOTS='reward=/path/to/reward_root florps=/path/to/florps_root' \
bash run_local_misalignment_eval_suite.sh
```

## Key Arguments

- `--tokenizer-id`
  Fallback tokenizer source for checkpoints without tokenizer files.
- `--batch-size`
  Local inference batch size.
- `--attn-implementation`
  Attention backend: `flash_attention_2`, `sdpa`, or `eager`.
- `--enable-thinking` / `--no-enable-thinking`
  Whether to request Qwen thinking mode where supported.
- `--reward-hacking-prompt-variant`
  Prompt framing for the adapted reward-hacking probes.
- `--betley-repeats`
- `--goals-samples`
- `--alignment-questions-limit`
- `--monitor-samples`
- `--exfil-samples`
- `--frame-samples`

## Outputs

By default, results are written under:

```text
results/misalignment_eval_suite/
```

The suite writes:
- detailed JSON bundle
- flat CSV and TSV of all raw rows
- per-eval summary CSV
- per-checkpoint summary CSV
- one plot per model family under `plots/`

## Metrics

### Malign Generalization Score

The suite computes a Malign Generalization Score (MGS) as the mean of:
- Betley
- Goals
- Alignment Questions
- Monitor Disruption
- Exfil Offer
- Frame Colleague

Each component is represented as a misalignment rate in `[0, 1]`, and MGS is the arithmetic mean of the available component scores.

### Reward-Hacking Rate

The reported `reward_hacking_rate` should be read as:
- how often the model emitted recognizable reward-hacking strategies under the adapted coding-task probes

It should not be read as:
- how often the model successfully hacked the original executed environment

## Provenance

The copied files under `src/misalignment_evals/` and `data/` are retained for transparency and provenance. The runtime path used by the local runner is lightweight and does not require `inspect_ai`, even though some copied source files still reflect the original implementation style.

## Limitations

- This is not a byte-for-byte reproduction of the original evaluation runtime stack.
- The reward-hacking metric is a tendency metric, not an execution-verified success metric.
- Misalignment judgments depend on a remote judge model and may vary with judge choice or prompt sensitivity.

## Intended Use

This suite is best suited for:
- comparative checkpoint analysis
- first-pass model screening
- reproducing the qualitative shape of misalignment-style evaluations on local models

It is not a drop-in substitute for:
- the original executed reward-hacking environments
- the exact full experimental stack from the source repository

If reward-hacking **success** rather than reward-hacking **tendency** is the target metric, integrate execution of the original coding environments and test harnesses.
