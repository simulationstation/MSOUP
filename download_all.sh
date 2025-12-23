#!/bin/bash
# =============================================================================
# BAO Analysis LSS Catalog Download Script
# =============================================================================
# This script downloads all required LSS catalogs for the preregistered
# environment-overlap BAO analysis with wedges and BAO template fits.
#
# Surveys: eBOSS DR16, DESI EDR, DESI DR1
# Tracers: LRGpCMASS, LRG, ELG, QSO
#
# Usage: ./download_all.sh [--eboss-only|--desi-edr-only|--desi-dr1-only|--all]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_FILE="${SCRIPT_DIR}/download.log"
CHECKSUM_FILE="${SCRIPT_DIR}/checksums_verified.txt"

# Base URLs
EBOSS_BASE="https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16"
DESI_EDR_BASE="https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
DESI_DR1_BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [[ -f "$dest" ]]; then
        log "${YELLOW}SKIP${NC}: $desc (already exists)"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    log "Downloading: $desc"

    if curl -L --progress-bar -o "$dest" "$url"; then
        log "${GREEN}OK${NC}: $desc"
        return 0
    else
        log "${RED}FAIL${NC}: $desc"
        rm -f "$dest"
        return 1
    fi
}

compute_checksum() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

verify_fits() {
    local file="$1"
    local name="$2"

    if python3 -c "
from astropy.io import fits
import sys
try:
    with fits.open('$file') as hdul:
        print(f'HDUs: {len(hdul)}')
        if len(hdul) > 1:
            print(f'Rows: {len(hdul[1].data)}')
            print(f'Columns: {list(hdul[1].columns.names)}')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1; then
        log "${GREEN}VERIFIED${NC}: $name"
        return 0
    else
        log "${RED}INVALID${NC}: $name"
        return 1
    fi
}

# =============================================================================
# eBOSS DR16 Downloads
# =============================================================================
download_eboss() {
    log "=========================================="
    log "Downloading eBOSS DR16 LSS Catalogs"
    log "=========================================="

    local EBOSS_LRGPCMASS_DIR="${DATA_DIR}/eboss/dr16/LRGpCMASS/v1"
    local EBOSS_QSO_DIR="${DATA_DIR}/eboss/dr16/QSO/v1"

    # LRGpCMASS Data catalogs
    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_data-NGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_data-NGC-vDR16.fits" \
        "eBOSS LRGpCMASS NGC data"

    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_data-SGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_data-SGC-vDR16.fits" \
        "eBOSS LRGpCMASS SGC data"

    # LRGpCMASS Random catalogs
    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_random-NGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_random-NGC-vDR16.fits" \
        "eBOSS LRGpCMASS NGC random"

    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_random-SGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_random-SGC-vDR16.fits" \
        "eBOSS LRGpCMASS SGC random"

    # LRGpCMASS Reconstructed catalogs (for BAO)
    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_data_rec-NGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_data_rec-NGC-vDR16.fits" \
        "eBOSS LRGpCMASS NGC data (reconstructed)"

    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_data_rec-SGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_data_rec-SGC-vDR16.fits" \
        "eBOSS LRGpCMASS SGC data (reconstructed)"

    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_random_rec-NGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_random_rec-NGC-vDR16.fits" \
        "eBOSS LRGpCMASS NGC random (reconstructed)"

    download_file "${EBOSS_BASE}/eBOSS_LRGpCMASS_clustering_random_rec-SGC-vDR16.fits" \
        "${EBOSS_LRGPCMASS_DIR}/eBOSS_LRGpCMASS_clustering_random_rec-SGC-vDR16.fits" \
        "eBOSS LRGpCMASS SGC random (reconstructed)"

    # QSO catalogs (alternate tracer for cross-check)
    download_file "${EBOSS_BASE}/eBOSS_QSO_clustering_data-NGC-vDR16.fits" \
        "${EBOSS_QSO_DIR}/eBOSS_QSO_clustering_data-NGC-vDR16.fits" \
        "eBOSS QSO NGC data"

    download_file "${EBOSS_BASE}/eBOSS_QSO_clustering_data-SGC-vDR16.fits" \
        "${EBOSS_QSO_DIR}/eBOSS_QSO_clustering_data-SGC-vDR16.fits" \
        "eBOSS QSO SGC data"

    download_file "${EBOSS_BASE}/eBOSS_QSO_clustering_random-NGC-vDR16.fits" \
        "${EBOSS_QSO_DIR}/eBOSS_QSO_clustering_random-NGC-vDR16.fits" \
        "eBOSS QSO NGC random"

    download_file "${EBOSS_BASE}/eBOSS_QSO_clustering_random-SGC-vDR16.fits" \
        "${EBOSS_QSO_DIR}/eBOSS_QSO_clustering_random-SGC-vDR16.fits" \
        "eBOSS QSO SGC random"

    log "eBOSS DR16 download complete"
}

# =============================================================================
# DESI EDR Downloads
# =============================================================================
download_desi_edr() {
    log "=========================================="
    log "Downloading DESI EDR v2.0 LSS Catalogs"
    log "=========================================="

    local DESI_LRG_DIR="${DATA_DIR}/desi/edr/LRG/v2.0"
    local DESI_ELG_DIR="${DATA_DIR}/desi/edr/ELG/v2.0"
    local DESI_QSO_DIR="${DATA_DIR}/desi/edr/QSO/v2.0"

    # LRG catalogs
    for region in N S; do
        download_file "${DESI_EDR_BASE}/LRG_${region}_clustering.dat.fits" \
            "${DESI_LRG_DIR}/LRG_${region}_clustering.dat.fits" \
            "DESI EDR LRG ${region} data"

        # Download first 4 random files (representative sample)
        for i in 0 1 2 3; do
            download_file "${DESI_EDR_BASE}/LRG_${region}_${i}_clustering.ran.fits" \
                "${DESI_LRG_DIR}/LRG_${region}_${i}_clustering.ran.fits" \
                "DESI EDR LRG ${region} random ${i}"
        done
    done

    # ELG catalogs (using ELGnotqso for cleaner sample)
    for region in N S; do
        download_file "${DESI_EDR_BASE}/ELGnotqso_${region}_clustering.dat.fits" \
            "${DESI_ELG_DIR}/ELGnotqso_${region}_clustering.dat.fits" \
            "DESI EDR ELGnotqso ${region} data"

        for i in 0 1 2 3; do
            download_file "${DESI_EDR_BASE}/ELGnotqso_${region}_${i}_clustering.ran.fits" \
                "${DESI_ELG_DIR}/ELGnotqso_${region}_${i}_clustering.ran.fits" \
                "DESI EDR ELGnotqso ${region} random ${i}"
        done
    done

    # QSO catalogs
    for region in N S; do
        download_file "${DESI_EDR_BASE}/QSO_${region}_clustering.dat.fits" \
            "${DESI_QSO_DIR}/QSO_${region}_clustering.dat.fits" \
            "DESI EDR QSO ${region} data"

        for i in 0 1 2 3; do
            download_file "${DESI_EDR_BASE}/QSO_${region}_${i}_clustering.ran.fits" \
                "${DESI_QSO_DIR}/QSO_${region}_${i}_clustering.ran.fits" \
                "DESI EDR QSO ${region} random ${i}"
        done
    done

    log "DESI EDR download complete"
}

# =============================================================================
# DESI DR1 Downloads
# =============================================================================
download_desi_dr1() {
    log "=========================================="
    log "Downloading DESI DR1 v1.5 LSS Catalogs"
    log "=========================================="

    local DESI_LRG_DIR="${DATA_DIR}/desi/dr1/LRG/v1.5"
    local DESI_ELG_DIR="${DATA_DIR}/desi/dr1/ELG/v1.5"
    local DESI_QSO_DIR="${DATA_DIR}/desi/dr1/QSO/v1.5"

    # LRG catalogs
    for region in NGC SGC; do
        download_file "${DESI_DR1_BASE}/LRG_${region}_clustering.dat.fits" \
            "${DESI_LRG_DIR}/LRG_${region}_clustering.dat.fits" \
            "DESI DR1 LRG ${region} data"

        # Download first 4 random files
        for i in 0 1 2 3; do
            download_file "${DESI_DR1_BASE}/LRG_${region}_${i}_clustering.ran.fits" \
                "${DESI_LRG_DIR}/LRG_${region}_${i}_clustering.ran.fits" \
                "DESI DR1 LRG ${region} random ${i}"
        done
    done

    # ELG catalogs (ELG_LOPnotqso is recommended for DR1)
    for region in NGC SGC; do
        download_file "${DESI_DR1_BASE}/ELG_LOPnotqso_${region}_clustering.dat.fits" \
            "${DESI_ELG_DIR}/ELG_LOPnotqso_${region}_clustering.dat.fits" \
            "DESI DR1 ELG_LOPnotqso ${region} data"

        for i in 0 1 2 3; do
            download_file "${DESI_DR1_BASE}/ELG_LOPnotqso_${region}_${i}_clustering.ran.fits" \
                "${DESI_ELG_DIR}/ELG_LOPnotqso_${region}_${i}_clustering.ran.fits" \
                "DESI DR1 ELG_LOPnotqso ${region} random ${i}"
        done
    done

    # QSO catalogs
    for region in NGC SGC; do
        download_file "${DESI_DR1_BASE}/QSO_${region}_clustering.dat.fits" \
            "${DESI_QSO_DIR}/QSO_${region}_clustering.dat.fits" \
            "DESI DR1 QSO ${region} data"

        for i in 0 1 2 3; do
            download_file "${DESI_DR1_BASE}/QSO_${region}_${i}_clustering.ran.fits" \
                "${DESI_QSO_DIR}/QSO_${region}_${i}_clustering.ran.fits" \
                "DESI DR1 QSO ${region} random ${i}"
        done
    done

    log "DESI DR1 download complete"
}

# =============================================================================
# Verification
# =============================================================================
verify_all() {
    log "=========================================="
    log "Verifying downloaded catalogs"
    log "=========================================="

    > "$CHECKSUM_FILE"

    find "$DATA_DIR" -name "*.fits" -type f | while read -r file; do
        local checksum=$(compute_checksum "$file")
        local relpath="${file#$SCRIPT_DIR/}"
        echo "$checksum  $relpath" >> "$CHECKSUM_FILE"
        verify_fits "$file" "$relpath"
    done

    log "Verification complete. Checksums written to: $CHECKSUM_FILE"
}

# =============================================================================
# Main
# =============================================================================
main() {
    log "Starting LSS catalog download"
    log "Data directory: $DATA_DIR"

    case "${1:-all}" in
        --eboss-only)
            download_eboss
            ;;
        --desi-edr-only)
            download_desi_edr
            ;;
        --desi-dr1-only)
            download_desi_dr1
            ;;
        --verify-only)
            verify_all
            ;;
        --all|*)
            download_eboss
            download_desi_edr
            download_desi_dr1
            verify_all
            ;;
    esac

    log "=========================================="
    log "Download complete"
    log "=========================================="
}

main "$@"
