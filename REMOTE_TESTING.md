# Remote Ubuntu testing

This checklist separates code validation from a paid model/Docker benchmark
run. Run commands from the repository root.

## 1. Clone and install

```bash
git clone <REPOSITORY_URL> mini-swe-agent-trace-r3
cd mini-swe-agent-trace-r3

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest ruff
```

Do not put API keys in this repository. Point mini-SWE-agent at the existing
task-specific configuration instead:

```bash
export MSWEA_GLOBAL_CONFIG_DIR="$HOME/CHKtask/.mini-swe-agent-config"
```

## 2. Offline code validation

These checks use no model API and no SWE-bench Docker image:

```bash
python -m ruff check src/minisweagent/trace_r3 tests/trace_r3

MSWEA_SILENT_STARTUP=1 \
MSWEA_GLOBAL_CONFIG_DIR="$PWD/.test-config" \
python -m pytest -q tests/trace_r3

mini-extra trace-r3 --help
```

The expected TRACE-R³ result for this revision is `24 passed`.

## 3. Docker and Hugging Face environment

For the existing rootless Docker installation:

```bash
export PATH="$HOME/bin:$PATH"
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
systemctl --user start docker.service

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DISABLE_XET=1

docker info --format \
  'Storage={{.Driver}} Root={{.DockerRootDir}} Security={{json .SecurityOptions}}'
```

The security options should include `name=rootless`.

If Docker Hub cannot pull a SWE-bench image, pull the same path through the
working proxy and tag it with the original name shown in the TRACE-R³ log:

```bash
ORIGINAL_IMAGE='docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-22080:latest'
PROXY_IMAGE='docker.1ms.run/swebench/sweb.eval.x86_64.sympy_1776_sympy-22080:latest'

docker pull "$PROXY_IMAGE"
docker tag "$PROXY_IMAGE" "$ORIGINAL_IMAGE"
docker image inspect "$ORIGINAL_IMAGE" --format '{{.Id}} {{.Size}}'
```

## 4. One-instance smoke experiment

Use one of the seven baseline failures first:

```bash
OUT="$HOME/CHKtask/experiments/v4flash_seed3318_trace_r3_smoke"

mini-extra trace-r3 \
  --subset verified \
  --split test \
  --filter '^sympy__sympy-22080$' \
  --model deepseek/deepseek-v4-flash \
  --environment-class docker \
  --workers 1 \
  --output "$OUT"
```

This is not the synthetic `python --version` smoke test. It runs the unchanged
mini-SWE-agent baseline in the real SWE-bench task image, applies Gate G0, and
activates TRACE-R³ only if the gate is not green.

## 5. Verify the result and all trajectories

```bash
python scripts/audit_trace_r3_run.py "$OUT"

find "$OUT" -type f \
  \( -name '*.traj.json' -o -name 'gate.json' -o -name 'input_graph.json' \
     -o -name 'rag_context.md' -o -name 'version_decision.json' \
     -o -name 'selection.json' \) \
  -print | sort
```

The audit exits non-zero if manifests, final selections, or expected
trajectories are inconsistent. A trajectory is atomically refreshed after
every agent interaction. An interrupted task retains its latest valid partial
trajectory and receives a new `attempt-NNN` directory when rerun.

## 6. Run or resume all seven failures

```bash
OUT="$HOME/CHKtask/experiments/v4flash_seed3318_trace_r3"
FILTER='^(sympy__sympy-22080|pytest-dev__pytest-10356|sympy__sympy-18763|sympy__sympy-17630|pytest-dev__pytest-5840|django__django-10973|sympy__sympy-16597)$'

mini-extra trace-r3 \
  --subset verified \
  --split test \
  --filter "$FILTER" \
  --model deepseek/deepseek-v4-flash \
  --environment-class docker \
  --workers 1 \
  --output "$OUT"
```

Run the same command after a disconnect or reboot. Completed records already
present in `preds.json` are skipped; interrupted records continue through a new
attempt directory. Use `--redo-existing` only when intentionally replacing a
completed prediction.

For a detached run:

```bash
tmux new-session -s trace3318
```

Start the seven-instance command inside that session, then detach with
`Ctrl-b`, followed by `d`. Reattach with:

```bash
tmux attach -t trace3318
```
