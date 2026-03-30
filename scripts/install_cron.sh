#!/usr/bin/env bash
# install_cron.sh — Install the AgentAugi nightly evolution crontab entry.
#
# Usage:
#   bash scripts/install_cron.sh            # Install (reads schedule from config)
#   bash scripts/install_cron.sh --remove   # Remove the crontab entry
#   bash scripts/install_cron.sh --dry-run  # Show what would be installed
#
# The script:
#   1. Resolves the repo root and Python interpreter.
#   2. Reads the cron schedule and timeout from configs/nightly_evolution.yaml.
#   3. Installs (or removes) a single crontab entry tagged with a unique comment.
#   4. Verifies the entry with `crontab -l`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRON_TAG="# agentaugi-nightly-evolution"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/configs/nightly_evolution.yaml"
EVOLUTION_SCRIPT="${REPO_ROOT}/scripts/nightly_evolution.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --remove    Remove the crontab entry instead of installing
  --dry-run   Print what would be installed without modifying crontab
  --help      Show this help message
EOF
}

info()    { echo "[INFO]  $*"; }
warn()    { echo "[WARN]  $*" >&2; }
error()   { echo "[ERROR] $*" >&2; }
die()     { error "$@"; exit 1; }

# Read a YAML scalar by key path (simple grep-based; no yq required).
# Usage: yaml_value "schedule.cron" "$CONFIG_FILE"
yaml_value() {
    local key="$1" file="$2"
    # Support one level of nesting: "section.key"
    local section key_name
    if [[ "$key" == *.* ]]; then
        section="${key%%.*}"
        key_name="${key#*.}"
        # Find the section block, then extract the key within it.
        awk "
            /^${section}:/ { in_section=1; next }
            in_section && /^[^ ]/ { in_section=0 }
            in_section && /^[[:space:]]+${key_name}:/ {
                gsub(/^[[:space:]]+${key_name}:[[:space:]]*/, \"\")
                gsub(/[\"']/, \"\")
                print; exit
            }
        " "$file"
    else
        grep -E "^${key}:" "$file" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'"'"
    fi
}

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------

MODE="install"  # install | remove | dry-run

for arg in "$@"; do
    case "$arg" in
        --remove)   MODE="remove"   ;;
        --dry-run)  MODE="dry-run"  ;;
        --help|-h)  usage; exit 0   ;;
        *) die "Unknown argument: $arg.  Run with --help for usage." ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve Python interpreter
# ---------------------------------------------------------------------------

find_python() {
    # Prefer the interpreter from the active virtualenv/conda, then system.
    for candidate in \
        "${VIRTUAL_ENV:-}/bin/python3" \
        "${CONDA_PREFIX:-}/bin/python3" \
        "$(command -v python3 2>/dev/null)" \
        "$(command -v python 2>/dev/null)"
    do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(find_python)" || die "No Python interpreter found.  Please activate a virtualenv."
info "Python interpreter: $PYTHON"

# ---------------------------------------------------------------------------
# Read schedule from config
# ---------------------------------------------------------------------------

CRON_SCHEDULE="0 2 * * *"    # sensible default
TIMEOUT_MINUTES="240"

if [[ -f "$CONFIG_FILE" ]]; then
    _sched="$(yaml_value "schedule.cron" "$CONFIG_FILE")"
    _timeout="$(yaml_value "schedule.timeout_minutes" "$CONFIG_FILE")"
    [[ -n "$_sched" ]]   && CRON_SCHEDULE="$_sched"
    [[ -n "$_timeout" ]] && TIMEOUT_MINUTES="$_timeout"
    info "Schedule from config: '$CRON_SCHEDULE'  timeout: ${TIMEOUT_MINUTES}m"
else
    warn "Config not found at ${CONFIG_FILE} — using default schedule '${CRON_SCHEDULE}'."
fi

# ---------------------------------------------------------------------------
# Build the crontab line
# ---------------------------------------------------------------------------

LOG_DIR="${REPO_ROOT}/data/evolution/logs"
LOG_FILE="${LOG_DIR}/nightly_evolution_\$(date +\\%Y\\%m\\%d_\\%H\\%M\\%S).log"

CRON_COMMAND="cd ${REPO_ROOT} && timeout $((TIMEOUT_MINUTES * 60)) ${PYTHON} ${EVOLUTION_SCRIPT} --config ${CONFIG_FILE} >> ${LOG_FILE} 2>&1"

CRON_LINE="${CRON_SCHEDULE}  ${CRON_COMMAND}  ${CRON_TAG}"

# ---------------------------------------------------------------------------
# Remove mode
# ---------------------------------------------------------------------------

remove_entry() {
    info "Removing crontab entry tagged '${CRON_TAG}' ..."
    local current
    current="$(crontab -l 2>/dev/null || true)"
    if ! echo "$current" | grep -qF "$CRON_TAG"; then
        info "No matching entry found — nothing to remove."
        return 0
    fi
    local updated
    updated="$(echo "$current" | grep -vF "$CRON_TAG")"
    echo "$updated" | crontab -
    info "Entry removed."
}

# ---------------------------------------------------------------------------
# Install mode
# ---------------------------------------------------------------------------

install_entry() {
    # Ensure log directory exists at job run-time by embedding mkdir in the cmd.
    # (The crontab line already wraps the script with `cd repo &&`)
    local mkdir_cmd="mkdir -p ${LOG_DIR}"
    CRON_LINE="${CRON_SCHEDULE}  ${mkdir_cmd} && ${CRON_COMMAND}  ${CRON_TAG}"

    info "Installing crontab entry ..."

    local current
    current="$(crontab -l 2>/dev/null || true)"

    # Remove any stale entry first (idempotent)
    local cleaned
    cleaned="$(echo "$current" | grep -vF "$CRON_TAG" || true)"

    # Append new entry (ensure trailing newline)
    local new_crontab
    if [[ -z "$cleaned" ]]; then
        new_crontab="${CRON_LINE}"$'\n'
    else
        new_crontab="${cleaned}"$'\n'"${CRON_LINE}"$'\n'
    fi

    echo "$new_crontab" | crontab -
    info "Crontab updated."
}

# ---------------------------------------------------------------------------
# Dry-run: just show what would happen
# ---------------------------------------------------------------------------

dry_run() {
    echo ""
    echo "=== DRY-RUN: would install the following crontab entry ==="
    echo ""
    local mkdir_cmd="mkdir -p ${LOG_DIR}"
    echo "${CRON_SCHEDULE}  ${mkdir_cmd} && ${CRON_COMMAND}  ${CRON_TAG}"
    echo ""
    echo "=== Current crontab ==="
    crontab -l 2>/dev/null || echo "(empty)"
}

# ---------------------------------------------------------------------------
# Verify the installed/removed entry
# ---------------------------------------------------------------------------

verify() {
    local mode="$1"
    local current
    current="$(crontab -l 2>/dev/null || true)"
    echo ""
    echo "=== Current crontab (after ${mode}) ==="
    if [[ -z "$current" ]]; then
        echo "(empty)"
    else
        echo "$current"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "$MODE" in
    remove)
        remove_entry
        verify "remove"
        ;;
    dry-run)
        dry_run
        ;;
    install)
        # Safety: confirm the script exists before installing
        if [[ ! -f "$EVOLUTION_SCRIPT" ]]; then
            die "Evolution script not found: ${EVOLUTION_SCRIPT}"
        fi
        install_entry
        verify "install"
        info "Done.  The nightly evolution will run: ${CRON_SCHEDULE}"
        info "Logs will appear in: ${LOG_DIR}/"
        ;;
esac
