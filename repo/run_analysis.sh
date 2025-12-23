#!/bin/bash
# ==============================================================================
# BAO Environment-Overlap Analysis - Main Execution Script
# ==============================================================================
# This script runs the complete analysis pipeline for eBOSS DR16 LRGpCMASS.
#
# Usage:
#   ./run_analysis.sh [--stage STAGE] [--dry-run] [--mocks-only] [--robustness]
#
# Stages:
#   0. setup      - Create environment, lock preregistration
#   1. io_qa      - Load catalogs, run QA sanity checks
#   2. density    - Build density field, compute E metric
#   3. correlation - Compute pair counts and correlation functions
#   4. bao_fit    - BAO template fits per E bin
#   5. inference  - Hierarchical beta inference (BLINDED)
#   6. mocks      - Run pipeline on mock catalogs
#   7. robustness - Run pre-declared robustness checks
#   8. unblind    - Unblind results (requires --confirm-unblind)
#   9. report     - Generate paper-ready figures and tables
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
CONFIG_DIR="${REPO_DIR}/configs"
OUTPUT_DIR="${REPO_DIR}/../today_results"
AUDIT_LOG="${OUTPUT_DIR}/audit_log.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "[${TIMESTAMP}] $1" | tee -a "${OUTPUT_DIR}/execution.log" 2>/dev/null || echo -e "[${TIMESTAMP}] $1"
}

log_audit() {
    local stage="$1"
    local status="$2"
    local details="${3:-}"

    mkdir -p "${OUTPUT_DIR}"

    python3 - <<EOF
import json
from pathlib import Path
from datetime import datetime

audit_file = Path("${AUDIT_LOG}")
entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "stage": "${stage}",
    "status": "${status}",
    "details": "${details}",
    "user": "$(whoami)",
    "host": "$(hostname)"
}

existing = []
if audit_file.exists():
    with open(audit_file) as f:
        existing = json.load(f)
existing.append(entry)
with open(audit_file, "w") as f:
    json.dump(existing, f, indent=2)
EOF
}

compute_prereg_hash() {
    sha256sum "${CONFIG_DIR}/preregistration.yaml" | awk '{print $1}'
}

check_blinding() {
    if [[ -f "${OUTPUT_DIR}/blinded_results.json" ]]; then
        local is_blinded=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/blinded_results.json'))['is_blinded'])")
        if [[ "$is_blinded" == "True" ]]; then
            echo "BLINDED"
        else
            echo "UNBLINDED"
        fi
    else
        echo "NOT_RUN"
    fi
}

# ==============================================================================
# Stage 0: Setup
# ==============================================================================
stage_setup() {
    log "${BLUE}[Stage 0] Setting up analysis environment${NC}"

    mkdir -p "${OUTPUT_DIR}/figures"
    mkdir -p "${OUTPUT_DIR}/data"
    mkdir -p "${OUTPUT_DIR}/robustness"

    # Compute and store preregistration hash
    PREREG_HASH=$(compute_prereg_hash)
    echo "${PREREG_HASH}" > "${OUTPUT_DIR}/prereg_hash.txt"
    log "Preregistration hash: ${PREREG_HASH}"

    # Check Python environment
    log "Checking Python environment..."
    python3 -c "import numpy, scipy, astropy; print('Core packages OK')"

    # Record environment
    pip freeze > "${OUTPUT_DIR}/environment.txt"
    python3 --version >> "${OUTPUT_DIR}/environment.txt"

    # Initialize blinding key if needed
    python3 -c "
from bao_overlap.blinding import BlindState, generate_blinding_key
from pathlib import Path
key_file = Path.home() / '.bao_blind_key'
if not key_file.exists():
    generate_blinding_key(key_file)
    print('Blinding key generated')
else:
    print('Blinding key exists')
"

    log_audit "setup" "complete" "prereg_hash=${PREREG_HASH}"
    log "${GREEN}[Stage 0] Setup complete${NC}"
}

# ==============================================================================
# Stage 1: IO and QA
# ==============================================================================
stage_io_qa() {
    log "${BLUE}[Stage 1] Loading catalogs and running QA${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage io_qa \
        --output "${OUTPUT_DIR}" \
        ${DRY_RUN:+--dry-run}

    log_audit "io_qa" "complete"
    log "${GREEN}[Stage 1] IO/QA complete${NC}"
}

# ==============================================================================
# Stage 2: Density Field
# ==============================================================================
stage_density() {
    log "${BLUE}[Stage 2] Building density field and computing E metric${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage density \
        --output "${OUTPUT_DIR}" \
        ${DRY_RUN:+--dry-run}

    log_audit "density" "complete"
    log "${GREEN}[Stage 2] Density field complete${NC}"
}

# ==============================================================================
# Stage 3: Correlation Functions
# ==============================================================================
stage_correlation() {
    log "${BLUE}[Stage 3] Computing correlation functions${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage correlation \
        --output "${OUTPUT_DIR}" \
        ${DRY_RUN:+--dry-run}

    log_audit "correlation" "complete"
    log "${GREEN}[Stage 3] Correlation functions complete${NC}"
}

# ==============================================================================
# Stage 4: BAO Template Fitting
# ==============================================================================
stage_bao_fit() {
    log "${BLUE}[Stage 4] BAO template fitting${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage bao_fit \
        --output "${OUTPUT_DIR}" \
        ${DRY_RUN:+--dry-run}

    log_audit "bao_fit" "complete"
    log "${GREEN}[Stage 4] BAO fitting complete${NC}"
}

# ==============================================================================
# Stage 5: Hierarchical Inference (BLINDED)
# ==============================================================================
stage_inference() {
    log "${BLUE}[Stage 5] Hierarchical beta inference (BLINDED)${NC}"
    log "${YELLOW}WARNING: Results will be blinded. Beta significance hidden.${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage inference \
        --output "${OUTPUT_DIR}" \
        --blinded \
        ${DRY_RUN:+--dry-run}

    log_audit "inference" "complete" "blinded=true"
    log "${GREEN}[Stage 5] Inference complete (BLINDED)${NC}"
}

# ==============================================================================
# Stage 6: Mock Analysis
# ==============================================================================
stage_mocks() {
    log "${BLUE}[Stage 6] Running mock analysis${NC}"

    local N_MOCKS=${N_MOCKS:-200}
    log "Processing ${N_MOCKS} mocks..."

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage mocks \
        --output "${OUTPUT_DIR}" \
        --n-mocks "${N_MOCKS}" \
        ${DRY_RUN:+--dry-run}

    log_audit "mocks" "complete" "n_mocks=${N_MOCKS}"
    log "${GREEN}[Stage 6] Mock analysis complete${NC}"
}

# ==============================================================================
# Stage 7: Robustness Checks
# ==============================================================================
stage_robustness() {
    log "${BLUE}[Stage 7] Running robustness checks${NC}"

    local variants=(
        "smoothing_R10"
        "smoothing_R20"
        "wedge_W025"
        "wedge_W03"
        "separation_S60_160"
        "separation_S40_200"
        "weight_NOTOP"
        "weight_NOCP"
        "fit_POLY1"
        "fit_POLY3"
        "binning_Q3"
        "binning_Q5"
    )

    for variant in "${variants[@]}"; do
        log "Running variant: ${variant}"

        python3 "${REPO_DIR}/scripts/run_stage.py" \
            --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
            --stage robustness \
            --output "${OUTPUT_DIR}/robustness/${variant}" \
            --variant "${variant}" \
            --blinded \
            ${DRY_RUN:+--dry-run}

        log_audit "robustness_${variant}" "complete"
    done

    # Generate robustness summary
    python3 -c "
import json
from pathlib import Path

results_dir = Path('${OUTPUT_DIR}/robustness')
summary = {'variants': [], 'pass_count': 0, 'fail_count': 0}

for variant_dir in sorted(results_dir.iterdir()):
    if variant_dir.is_dir():
        result_file = variant_dir / 'robustness_result.json'
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            summary['variants'].append({
                'name': variant_dir.name,
                'passed': result.get('passed', False),
                'beta_change': result.get('beta_change_sigma', None)
            })
            if result.get('passed', False):
                summary['pass_count'] += 1
            else:
                summary['fail_count'] += 1

with open('${OUTPUT_DIR}/robustness_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f\"Robustness: {summary['pass_count']}/{len(summary['variants'])} passed\")
"

    log_audit "robustness" "complete"
    log "${GREEN}[Stage 7] Robustness checks complete${NC}"
}

# ==============================================================================
# Stage 8: Unblinding
# ==============================================================================
stage_unblind() {
    log "${BLUE}[Stage 8] Unblinding${NC}"

    if [[ "${CONFIRM_UNBLIND:-false}" != "true" ]]; then
        log "${RED}ERROR: Unblinding requires --confirm-unblind flag${NC}"
        log "Ensure all robustness checks are complete before unblinding."
        exit 1
    fi

    # Check robustness completion
    if [[ ! -f "${OUTPUT_DIR}/robustness_summary.json" ]]; then
        log "${RED}ERROR: Robustness checks not complete. Run stage 7 first.${NC}"
        exit 1
    fi

    python3 "${REPO_DIR}/scripts/unblind.py" \
        --results "${OUTPUT_DIR}/blinded_results.json" \
        --output "${OUTPUT_DIR}/unblinded_results.json" \
        --audit-log "${AUDIT_LOG}" \
        --confirm

    log_audit "unblind" "complete"
    log "${GREEN}[Stage 8] Unblinding complete${NC}"
}

# ==============================================================================
# Stage 9: Reporting
# ==============================================================================
stage_report() {
    log "${BLUE}[Stage 9] Generating reports${NC}"

    python3 "${REPO_DIR}/scripts/run_stage.py" \
        --config "${CONFIG_DIR}/runs/eboss_lrgpcmass_default.yaml" \
        --stage report \
        --output "${OUTPUT_DIR}"

    log_audit "report" "complete"
    log "${GREEN}[Stage 9] Reporting complete${NC}"

    log "Final artifacts:"
    ls -la "${OUTPUT_DIR}/"
    ls -la "${OUTPUT_DIR}/figures/" 2>/dev/null || true
}

# ==============================================================================
# Main
# ==============================================================================
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --stage STAGE       Run specific stage (0-9 or name)"
    echo "  --dry-run           Run with reduced data for testing"
    echo "  --mocks-only        Run only mock analysis"
    echo "  --robustness        Run only robustness checks"
    echo "  --confirm-unblind   Required for unblinding stage"
    echo "  --n-mocks N         Number of mocks to process (default: 200)"
    echo "  --help              Show this help"
    echo ""
    echo "Stages:"
    echo "  0/setup       - Create environment, lock preregistration"
    echo "  1/io_qa       - Load catalogs, run QA"
    echo "  2/density     - Build density field, compute E"
    echo "  3/correlation - Compute correlation functions"
    echo "  4/bao_fit     - BAO template fits"
    echo "  5/inference   - Hierarchical beta inference (BLINDED)"
    echo "  6/mocks       - Mock catalog analysis"
    echo "  7/robustness  - Pre-declared robustness checks"
    echo "  8/unblind     - Unblind results"
    echo "  9/report      - Generate paper-ready outputs"
}

main() {
    local STAGE=""
    DRY_RUN=""
    CONFIRM_UNBLIND=""
    N_MOCKS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage)
                STAGE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="--dry-run"
                shift
                ;;
            --confirm-unblind)
                CONFIRM_UNBLIND="true"
                shift
                ;;
            --n-mocks)
                N_MOCKS="$2"
                shift 2
                ;;
            --mocks-only)
                STAGE="mocks"
                shift
                ;;
            --robustness)
                STAGE="robustness"
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    export DRY_RUN CONFIRM_UNBLIND N_MOCKS

    log "========================================"
    log "BAO Environment-Overlap Analysis"
    log "========================================"
    log "Timestamp: ${TIMESTAMP}"
    log "Blinding status: $(check_blinding)"

    if [[ -n "$STAGE" ]]; then
        case $STAGE in
            0|setup) stage_setup ;;
            1|io_qa) stage_io_qa ;;
            2|density) stage_density ;;
            3|correlation) stage_correlation ;;
            4|bao_fit) stage_bao_fit ;;
            5|inference) stage_inference ;;
            6|mocks) stage_mocks ;;
            7|robustness) stage_robustness ;;
            8|unblind) stage_unblind ;;
            9|report) stage_report ;;
            *)
                echo "Unknown stage: $STAGE"
                print_usage
                exit 1
                ;;
        esac
    else
        # Run all stages except unblind
        stage_setup
        stage_io_qa
        stage_density
        stage_correlation
        stage_bao_fit
        stage_inference
        stage_mocks
        stage_robustness
        log "${YELLOW}All blinded stages complete. Run with --stage unblind --confirm-unblind to unblind.${NC}"
    fi

    log "========================================"
    log "Analysis complete"
    log "========================================"
}

main "$@"
