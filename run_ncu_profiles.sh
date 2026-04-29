#!/bin/bash
# run_ncu_profiles.sh — Batch NCU profiling for depthwise conv kernels
#
# Profiles all 3 implementations (custom, cuDNN-nchw, cuDNN-nhwc, triton-nhwc)
# across 6 configs. Uses 6 NCU metric sections per profile.
#
# Total: ~18 profiles, estimated 60-90 minutes.
#
# Usage:
#   chmod +x run_ncu_profiles.sh
#   ./run_ncu_profiles.sh                  # run all
#   ./run_ncu_profiles.sh --dry-run        # print commands only
#   ./run_ncu_profiles.sh --only custom    # only custom kernels
#   ./run_ncu_profiles.sh --only cudnn     # only cuDNN kernels
#   ./run_ncu_profiles.sh --only triton    # only Triton kernels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/ncu"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_ncu.py"
CONDA_ENV="chConv"
NCU="ncu"

# NCU settings
WARMUP_SKIP=10
LAUNCH_COUNT=5
NCU_SECTIONS=(
    "SpeedOfLight_HierarchicalHalfRooflineChart"
    "MemoryWorkloadAnalysis"
    "Occupancy"
    "ComputeWorkloadAnalysis"
    "WarpStateStats"
)

# Build section flags for NCU command
SECTION_FLAGS=""
for sec in "${NCU_SECTIONS[@]}"; do
    SECTION_FLAGS="${SECTION_FLAGS} --section ${sec}"
done

mkdir -p "$RESULTS_DIR"

# ---- config definitions ------------------------------------------------------

# All 6 configs
ALL_CONFIGS=(
    "9x9_1536_32"
    "7x7_768_32"
    "31x31_128_56"
    "29x29_256_28"
    "27x27_512_14"
    "13x13_1024_7"
)

# Configs with custom kernels (no custom kernel for 13x13)
CUSTOM_CONFIGS=(
    "9x9_1536_32"
    "7x7_768_32"
    "31x31_128_56"
    "29x29_256_28"
    "27x27_512_14"
)

# RepLKNet configs (for cuDNN NHWC comparison)
REPLKNET_CONFIGS=(
    "31x31_128_56"
    "29x29_256_28"
    "27x27_512_14"
    "13x13_1024_7"
)

# ConvMixer configs (for Triton, since Triton collapses on RepLKNet)
CONVMIXER_CONFIGS=(
    "9x9_1536_32"
    "7x7_768_32"
)

# Custom kernel name patterns (for --kernel-name filtering)
declare -A CUSTOM_KERNEL_NAMES
CUSTOM_KERNEL_NAMES["9x9_1536_32"]="depthwise_32_1536_9_kernel"
CUSTOM_KERNEL_NAMES["7x7_768_32"]="depthwise_32_768_7_kernel"
CUSTOM_KERNEL_NAMES["31x31_128_56"]="depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12"
CUSTOM_KERNEL_NAMES["29x29_256_28"]="depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12"
CUSTOM_KERNEL_NAMES["27x27_512_14"]="depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64"

# ---- helper functions --------------------------------------------------------

run_ncu() {
    local label="$1"
    local kernel_pattern="$2"
    local config="$3"
    local method="$4"
    local csv_out="${RESULTS_DIR}/${label}.csv"

    echo "=== $(date '+%H:%M:%S')  Profiling: ${label} ==="
    echo "    Config: ${config}  Method: ${method}"
    echo "    Output: ${csv_out}"

    if [ "${DRY_RUN:-false}" = "true" ]; then
        echo "    [DRY RUN] ncu --set full --section ... --csv --log-file ${csv_out} ..."
        return
    fi

    $NCU --set full \
        $SECTION_FLAGS \
        --target-processes all \
        --kernel-name "$kernel_pattern" \
        --launch-skip "$WARMUP_SKIP" \
        --launch-count "$LAUNCH_COUNT" \
        --csv --log-file "$csv_out" \
        --force-overwrite \
        conda run -n "$CONDA_ENV" python "$PROFILE_SCRIPT" \
            --config "$config" --method "$method" \
            --warmup 20 --iterations 30 \
            2>&1 | tail -3

    echo "    Done: $(wc -l < "$csv_out") lines in CSV"
    echo ""
}

# ---- main --------------------------------------------------------------------

DRY_RUN=false
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "NCU Profiling Campaign"
echo "============================================"
echo "Sections: ${NCU_SECTIONS[*]}"
echo "Warmup skip: ${WARMUP_SKIP}  Launch count: ${LAUNCH_COUNT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Dry run: ${DRY_RUN}"
echo "Only: ${ONLY:-all}"
echo ""

# ---- Custom kernels (5 profiles) ---------------------------------------------

if [ -z "$ONLY" ] || [ "$ONLY" = "custom" ]; then
    echo "--- Custom Kernels (5 profiles) ---"
    for config in "${CUSTOM_CONFIGS[@]}"; do
        kernel_name="${CUSTOM_KERNEL_NAMES[$config]}"
        label="custom_${config}"
        run_ncu "$label" "$kernel_name" "$config" "custom"
    done
fi

# ---- cuDNN NCHW (6 profiles, best per-kernel layout) ------------------------

if [ -z "$ONLY" ] || [ "$ONLY" = "cudnn" ]; then
    echo "--- cuDNN NCHW (6 profiles, best per-kernel layout) ---"
    for config in "${ALL_CONFIGS[@]}"; do
        label="cudnn_nchw_${config}"
        # cuDNN NCHW uses internal cuDNN kernels — capture all
        run_ncu "$label" "regex:.*" "$config" "cudnn_nchw"
    done

    echo "--- cuDNN NHWC (4 profiles, network-deployed layout) ---"
    for config in "${REPLKNET_CONFIGS[@]}"; do
        label="cudnn_nhwc_${config}"
        run_ncu "$label" "regex:.*" "$config" "cudnn_nhwc"
    done
fi

# ---- Triton NHWC (2 profiles, ConvMixer configs only) -----------------------

if [ -z "$ONLY" ] || [ "$ONLY" = "triton" ]; then
    echo "--- Triton NHWC (2 profiles) ---"
    for config in "${CONVMIXER_CONFIGS[@]}"; do
        label="triton_nhwc_${config}"
        # Triton JIT kernels contain function name pattern
        run_ncu "$label" "regex:.*depthwise.*" "$config" "triton_nhwc"
    done
fi

echo "=== $(date '+%H:%M:%S')  All profiles complete ==="
echo "Results in: ${RESULTS_DIR}/"
ls -la "$RESULTS_DIR/"