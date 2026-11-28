#!/usr/bin/env bash
# Vast.ai Linux instance. Paths are relative to the extracted kit except for the shared model/venv.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
kit="$PWD"
out="$kit/results"
dedicated_env="${BBH_ENV:-/workspace/bbh-mechanisms-v1-env}"
export HF_HOME="${BBH_HF_CACHE:-${HF_HOME:-/workspace/bbh-model-cache}}"
# Respect the instance's device mapping; the code uses cuda:0.
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
action="${1:-help}"
if (($#)); then shift; fi
die() { echo "$*" >&2; exit 1; }
usage() {
    echo "bash study.sh setup | access | plan | preflight | smoke | start [pilot|all|main|steps|params|random|controls]"
    echo "bash study.sh status [stage] | stop | analyze | archive"
    echo "bash study.sh run [stage]   # foreground"
}
if [[ "$action" == help ]]; then usage; exit; fi

# Jobs/status/analysis can be inspected from a plain Python 3.10+ shell.
base_py=$(command -v python3 || command -v python || true)
[[ -n "$base_py" ]] || die "Install Python 3 to use the kit."
case "$action" in
    plan|status|stop)
        exec "$base_py" driver.py "$action" "$@" ;;
    analyze)
        mkdir -p "$out"
        exec flock -n "$out/.analysis.lock" "$base_py" analysis.py --root "$out" "$@" ;;
    archive)
        [[ -d "$out" ]] || die "No results directory to archive"
        (
            flock -n -s 8 && flock -n -s 9 && flock -n -s 7 || die "A driver or analysis is active. Wait for it to finish, then archive."
            destination="${BBH_ARCHIVES:-/workspace}"
            mkdir -p "$destination"
            bundle=$(mktemp "$destination/BBH_Mechanisms_XXXXXXXX.tar.gz")
            trap 'rm -f -- "$bundle" "$bundle.sha256"' ERR
            # Only distributed source and output directories are eligible. Keep credentials, envs and weights external.
            shopt -s nullglob
            sources=(*.py *.sh *.md *.txt *.sha256 bbh_eval data)
            tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.lock' \
                --exclude='results/.stop*' -czf "$bundle" "${sources[@]}" results
            tar -tzf "$bundle" >/dev/null
            (cd "$destination" && sha256sum "$(basename -- "$bundle")") > "$bundle.sha256"
            trap - ERR
            echo "Download both files before ending the rental:"
            echo "$bundle"
            echo "$bundle.sha256"
        ) 7>>"$out/.analysis.lock" 8>>"$out/.launch.lock" 9>>"$out/.driver.lock"
        exit ;;
esac

state="$kit/.runtime-python"
choose_python() {
    if [[ -n "${BBH_PYTHON:-}" ]]; then
        [[ -x "$BBH_PYTHON" ]] || die "BBH_PYTHON must point to an executable Python"
        py="$BBH_PYTHON"
    elif [[ -f "$state" ]] && [[ -x "$(cat "$state")" ]]; then
        py=$(cat "$state")
    else
        py="$dedicated_env/bin/python"
    fi
}
if [[ "$action" == setup ]]; then
    mkdir -p "$out/provenance"
    if [[ -f checksums.sha256 ]]; then sha256sum --quiet -c checksums.sha256; fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    df -h "$kit"
    choose_python
    if [[ -n "${BBH_PYTHON:-}" ]]; then
        "$py" runtime.py packages
    else
        selected=""
        for candidate in "$py" /workspace/bbh-controlled-v2-env/bin/python /workspace/bbh-a100-env/bin/python \
            /venv/main/bin/python "$(command -v python3.10 || true)"; do
            if [[ -x "$candidate" ]] && "$candidate" runtime.py packages >/dev/null 2>&1; then
                selected="$candidate"
                break
            fi
        done
        if [[ -n "$selected" ]]; then
            py="$selected"
        else
            command -v python3.10 >/dev/null || die "Choose the Vast CUDA 12.1 / Python 3.10 template or point BBH_PYTHON to the prior setup."
            if [[ ! -x "$dedicated_env/bin/python" ]]; then python3.10 -m venv "$dedicated_env"; fi
            py="$dedicated_env/bin/python"
            "$py" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu121 'torch==2.4.0+cu121'
            "$py" -m pip install --disable-pip-version-check \
                'transformers==4.47.1' 'accelerate==1.1.1' 'huggingface-hub==0.36.0' 'numpy==1.26.4' \
                'safetensors==0.8.0' 'tokenizers==0.21.4'
        fi
    fi
    "$py" runtime.py packages
    "$py" runtime.py gpu
    "$py" -m pip check
    "$py" -m unittest -v bbh_eval.test_llama_cache
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    # pip list excludes credential-bearing direct URLs which pip freeze could include.
    "$py" -m pip list --format=json > "$out/provenance/packages-$stamp.json"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$out/provenance/gpu-$stamp.csv"
    printf '%s\n' "$py" > "$state"
    echo "Environment ready: $py"
    exit
fi
choose_python
[[ -x "$py" ]] || die "Run bash study.sh setup first."
if [[ "$action" == access ]]; then
    exec "$py" runtime.py access
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$action" in
    preflight) exec "$py" runtime.py preflight ;;
    run|smoke|start)
        if [[ "$action" == smoke ]]; then (($# == 0)) || die "smoke takes no stage"; fi
        (($# <= 1)) || die "Supply at most one stage; use bash study.sh help"
        stage="${1:-all}"
        case "$stage" in pilot|all|main|steps|params|random|controls) ;; *) die "Invalid stage: $stage";; esac
        model=$("$py" runtime.py path)
        if [[ "$action" == smoke ]]; then
            exec "$py" -u driver.py smoke --model "$model"
        elif [[ "$action" == run ]]; then
            exec "$py" -u driver.py run "$stage" --model "$model"
        fi
        mkdir -p "$out/driver-logs"
        (
            flock -n 8 || die "A background driver is launching or active; use status."
            flock -n 9 || die "The experiment is already active; use status."
            logfile=$(mktemp "$out/driver-logs/${stage}-XXXXXXXX.log")
            # Child inherits the launch lock until it exits. Release the preflight lock
            # before exec so that even a fast driver can acquire its own .driver.lock.
            flock -u 9
            nohup "$py" -u driver.py run "$stage" --model "$model" 9>&- >"$logfile" 2>&1 </dev/null &
            child_pid=$!
            echo "Started PID $child_pid; log: $logfile"
        ) 8>>"$out/.launch.lock" 9>>"$out/.driver.lock"
        ;;
    *) usage; die "Unknown action: $action" ;;
esac
