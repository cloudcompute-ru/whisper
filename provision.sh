#!/usr/bin/env bash
#
# Provision script for Whisper WebUI (slug: whisper).
# Stages: install_runtime, download_model, start_server (port 7860).
#
set -Eeuo pipefail

CC_PROVISION_URL="${CC_PROVISION_URL:-}"
CC_AGENT_TOKEN="${CC_AGENT_TOKEN:-}"
WHISPER_PORT="${WHISPER_PORT:-7860}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
APP_DIR="${APP_DIR:-/workspace/cc-whisper}"
VENV_DIR="${APP_DIR}/venv"
PYVERSION="${CC_WHISPER_PYTHON_VERSION:-3.11}"
CURRENT_STAGE="install_runtime"

report_stage() {
    if [ -z "$CC_PROVISION_URL" ] || [ -z "$CC_AGENT_TOKEN" ]; then
        return 0
    fi
    curl -fsS -X POST "$CC_PROVISION_URL" \
        -H "Authorization: Bearer $CC_AGENT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$1" --max-time 5 >/dev/null 2>&1 || true
}

log() { echo "[cc-provision] $*"; }

send_log_tail() {
    if [ -z "$CC_PROVISION_URL" ] || [ -z "$CC_AGENT_TOKEN" ]; then return 0; fi
    local encoded
    encoded="$(tail -n 200 /var/log/cc-provision.log 2>/dev/null \
        | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null)" || return 0
    [ -z "$encoded" ] && return 0
    report_stage "{\"stage\":\"${CURRENT_STAGE}\",\"log_tail\":${encoded}}"
}

setup_python() {
    if [ -x "${VENV_DIR}/bin/python" ]; then
        PY="${VENV_DIR}/bin/python"
        PIP="${VENV_DIR}/bin/pip"
        return 0
    fi
    mkdir -p "$APP_DIR"
    if ! command -v uv >/dev/null 2>&1; then
        curl -fsSL https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    uv venv --python "$PYVERSION" "$VENV_DIR"
    PY="${VENV_DIR}/bin/python"
    PIP="${VENV_DIR}/bin/pip"
}

# --- install_runtime -------------------------------------------------------

log "stage: install_runtime"
report_stage '{"stage":"install_runtime","progress_pct":0}'

setup_python
"$PIP" install --no-cache-dir -U pip wheel
"$PIP" install --no-cache-dir \
    "faster-whisper==1.1.1" \
    "gradio==5.23.0" \
    "torch" \
    "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

report_stage '{"stage":"install_runtime","progress_pct":100}'
send_log_tail

# --- download_model --------------------------------------------------------

CURRENT_STAGE="download_model"
log "stage: download_model"
report_stage '{"stage":"download_model","progress_pct":0}'

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
"$PY" - <<PY
from faster_whisper import WhisperModel
WhisperModel("${WHISPER_MODEL}", device="cuda", compute_type="float16")
print("model ready", flush=True)
PY

report_stage '{"stage":"download_model","progress_pct":100}'
send_log_tail

# --- start_server ----------------------------------------------------------

CURRENT_STAGE="start_server"
log "stage: start_server"
report_stage '{"stage":"start_server"}'

cat > "${APP_DIR}/app.py" <<'PYAPP'
import gradio as gr
from faster_whisper import WhisperModel

MODEL = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe(audio):
    if audio is None:
        return "Upload an audio file."
    segments, info = MODEL.transcribe(audio, beam_size=5)
    text = " ".join(s.text.strip() for s in segments)
    return f"[{info.language}] {text}"

demo = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs="text", title="Whisper")
demo.launch(server_name="0.0.0.0", server_port=int(__import__("os").environ.get("WHISPER_PORT", "7860")))
PYAPP

export WHISPER_PORT
nohup "$PY" "${APP_DIR}/app.py" > /var/log/cc-whisper.log 2>&1 &
APP_PID=$!

BIND_TIMEOUT_S=90
for _ in $(seq 1 "$BIND_TIMEOUT_S"); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${WHISPER_PORT}/" >/dev/null 2>&1; then
        send_log_tail
        report_stage '{"stage":"start_server","progress_pct":100}'
        log "provisioning complete"
        exit 0
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        tail_msg="$(tail -c 500 /var/log/cc-whisper.log 2>/dev/null | tr -d '\r' | tr '\n' ' ' | sed 's/"/'"'"'/g')"
        send_log_tail
        report_stage "{\"stage\":\"start_server\",\"message\":\"Whisper UI crashed: ${tail_msg}\"}"
        exit 1
    fi
    sleep 1
done

report_stage "{\"stage\":\"start_server\",\"message\":\"Whisper UI not ready in ${BIND_TIMEOUT_S}s.\"}"
exit 1
