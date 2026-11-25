#!/usr/bin/env bash
# One Vast.ai A100. All operations are repeatable in a fresh Jupyter terminal.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
kit="$PWD"
out="$kit/results"
envdir="${BBH_ENV:-/workspace/bbh-controlled-v2-env}"
export HF_HOME="${BBH_HF_CACHE:-/workspace/bbh-model-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
manifest="$kit/data/manifest.json"
source_dir="$kit/data/bbh"
action="${1:-help}"
if (($#)); then shift; fi

fail() { echo "$*" >&2; exit 1; }
if [[ "$action" == init ]]; then
    sha256sum --quiet -c checksums.sha256
    mkdir -- "$out" || fail "Results already exist. To resume this experiment use status and start. For a new experiment extract the kit in a new folder."
    mkdir "$out/provenance"
    date -u +%FT%TZ > "$out/provenance/fresh-start-utc.txt"
    echo "Created an empty experiment at $out"
    exit 0
fi
if [[ "$action" != help ]]; then
    [[ -f "$out/provenance/fresh-start-utc.txt" ]] || fail 'Run bash study.sh init in a newly extracted kit.'
fi
python_env() {
    if [[ -n "${BBH_PYTHON:-}" ]]; then
        py="$BBH_PYTHON"
    elif [[ -x "$envdir/bin/python" ]]; then
        py="$envdir/bin/python"
    elif [[ -x /workspace/bbh-a100-env/bin/python ]]; then
        py=/workspace/bbh-a100-env/bin/python
    else
        py="$envdir/bin/python"
    fi
}

runtime_check() {
    "$py" - <<'PY'
import sys
from importlib.metadata import version
import torch
assert sys.version_info[:2] == (3, 10), sys.version
expected = {'transformers':'4.47.1','accelerate':'1.1.1','huggingface-hub':'0.36.0','numpy':'1.26.4'}
for name, v in expected.items():
    assert version(name) == v, (name, version(name), v)
assert torch.__version__.split('+')[0] == '2.4.0' and torch.version.cuda == '12.1', torch.__version__
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA or BF16 unavailable'
gpu = torch.cuda.get_device_properties(0)
assert 'A100' in gpu.name and gpu.total_memory >= 78 * 2**30, (gpu.name, gpu.total_memory)
print('Verified', gpu.name, round(gpu.total_memory/2**30,1), 'GiB', torch.__version__, torch.version.cuda)
PY
}

if [[ "$action" == setup ]]; then
    sha256sum --quiet -c checksums.sha256
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    df -h "$kit"
    python_env
    if ! [[ -x "$py" ]] || ! runtime_check; then
        if [[ -n "${BBH_PYTHON:-}" ]]; then fail 'BBH_PYTHON does not match the pinned runtime.'; fi
        command -v python3.10 >/dev/null || fail 'Use the Vast Python 3.10 / CUDA 12.1 image.'
        if [[ ! -x "$envdir/bin/python" ]]; then python3.10 -m venv "$envdir"; fi
        py="$envdir/bin/python"
        "$py" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu121 'torch==2.4.0'
        "$py" -m pip install --disable-pip-version-check 'transformers==4.47.1' 'accelerate==1.1.1' 'huggingface-hub==0.36.0' 'numpy==1.26.4'
    fi
    runtime_check
    "$py" -m pip check
    mkdir -p "$out/provenance"
    if [[ -f "$out/provenance/manifest.json" ]]; then cmp "$manifest" "$out/provenance/manifest.json"; fi
    cp "$manifest" "$out/provenance/manifest.json"
    cp checksums.sha256 "$out/provenance/kit-sha256.txt"
    nvidia-smi -q > "$out/provenance/gpu-$(date -u +%Y%m%dT%H%M%SZ).txt"
    "$py" -m pip freeze > "$out/provenance/packages-$(date -u +%Y%m%dT%H%M%SZ).txt"
    "$py" -m unittest -v bbh_eval.test_llama_cache
    exit
fi

python_env
case "$action" in
    access)
        # Interactive masked input through huggingface_hub only. Never put a token on the command line.
        [[ -x "$py" ]] || fail 'Run study.sh setup first.'
        "$py" - <<'PY'
from getpass import getpass
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
from bbh_eval.run_llama_icl import MODEL, MODEL_REV
def fetch(**kwargs):
    return Path(snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=[
        'config.json', 'generation_config.json', 'tokenizer*.json', 'special_tokens_map.json',
        'model*.safetensors', 'model.safetensors.index.json'], **kwargs))
def complete(p):
    required = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']
    if not all((p / name).is_file() for name in required):
        return False
    shards = set(json.loads((p / required[-1]).read_text())['weight_map'].values())
    return bool(shards) and all((p / name).is_file() for name in shards)
try:
    p = fetch(local_files_only=True)
except LocalEntryNotFoundError:
    p = None
if p is None or not complete(p):
    try:
        p = fetch()
    except HfHubHTTPError as exc:
        if not isinstance(exc, GatedRepoError) and getattr(exc.response, 'status_code', None) not in (401, 403):
            raise
        p = fetch(token=getpass('Paste your approved Hugging Face read token (hidden): ').strip())
if not complete(p):
    raise SystemExit('Download incomplete: required config, tokenizer or weight shards missing.')
print('Model and tokenizer cached at', p)
PY
        ;;
    plan|status|run)
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        if [[ "$action" == run ]]; then
            [[ -x "$py" ]] || fail 'Run study.sh setup first.'
            runtime_check
        else
            [[ -x "$py" ]] || py=python3
        fi
        exec "$py" -u run_matrix.py "$action" --manifest "$manifest" --source-dir "$source_dir" --root "$out" "$@"
        ;;
    precheck)
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        exec "$py" precheck.py --manifest "$manifest" --source-dir "$source_dir" --out "$out/preflight.csv" "$@"
        ;;
    analyze)
        [[ -x "$py" ]] || py=python3
        exec "$py" analyze_rerun.py --manifest "$manifest" --root "$out" "$@"
        ;;
    start)
        mkdir -p "$out/driver-logs"
        (
            # This lock follows the launched process from setup checks through its last worker.
            flock -n 8 || fail 'This study already has an active driver. Use status or pause.'
            flock -n 9 || fail 'Another matrix process holds the output folder.'
            log=$(mktemp "$out/driver-logs/formal-XXXXXXXX.log")
            rm -f -- "$out/.pause-after-current"
            nohup bash "$kit/study.sh" run --continue-on-oom "$@" >"$log" 2>&1 </dev/null 9>&- &
            echo "Launched PID=$!; driver log: $log"
        ) 8>>"$out/.launch.lock" 9>>"$out/.matrix.lock"
        ;;
    pause)
        [[ -d "$out" ]] || fail "No result folder: $out"
        touch "$out/.pause-after-current"
        echo 'Pause requested. The driver will exit after the current configuration.'
        ;;
    finish)
        [[ -x "$py" ]] || py=python3
        (
            flock -n -s 8 && flock -n -s 9 || fail 'Experiment is active. Check status, then finish after the driver exits.'
            bash "$kit/study.sh" analyze
            "$py" "$kit/shot_effects.py" --analysis "$out/analysis-shots-two"
            bash "$kit/study.sh" archive
        ) 8>>"$out/.launch.lock" 9>>"$out/.matrix.lock"
        ;;
    archive)
        [[ -d "$out/runs" ]] || fail "No formal runs under $out"
        destination="${BBH_ARCHIVES:-/workspace}"
        mkdir -p "$destination"
        bundle=$(mktemp "$destination/BBH_Attention_144_XXXXXXXX.tar.gz")
        # Refuse a racing writer. Formal output only: model cache and token live outside this folder.
        (
            flock -n -s 9 || fail 'A formal job is running. Archive after it has stopped.'
            tar -czf "$bundle" -C "$(dirname "$out")" "$(basename "$out")"
        ) 9>>"$out/.matrix.lock" || { rm -f "$bundle"; exit 1; }
        tar -tzf "$bundle" >/dev/null
        (cd "$destination" && sha256sum "$(basename "$bundle")") > "$bundle.sha256"
        echo "$bundle"; echo "$bundle.sha256"
        ;;
    help)
        echo 'bash study.sh init | setup | access | precheck | plan | start | status | pause | analyze | finish | archive'
        ;;
    *)
        fail "Unknown action: $action"
        ;;
esac
