#!/usr/bin/env bash
# Source this file to make python3.11 + torch (cu121 wheel) find NVIDIA shared libs
# AND select the CUDA visible device on the T4 host.
#
# Usage:
#   source scripts/cuda_env.sh
#   python3.11 experiments/benchmark_single_monomial.py ...
#
# Path layout discovered on the host (2026-06-02):
#   - python3.11 binary:    /usr/bin/python3.11
#   - torch + nvidia wheels: /usr/local/lib/python3.11/site-packages/nvidia/<cmp>/lib
#   - GPU:                  Tesla T4 (CUDA 12.2 driver, 12.1 cu wheel)
#
# This script is committed so any future tool / experiment in apolarity/ can
# rely on a single canonical entry point for the T4 environment.

_NV_BASE=/usr/local/lib/python3.11/site-packages/nvidia

export LD_LIBRARY_PATH="\
${_NV_BASE}/cudnn/lib:\
${_NV_BASE}/cuda_runtime/lib:\
${_NV_BASE}/cuda_nvrtc/lib:\
${_NV_BASE}/cuda_cupti/lib:\
${_NV_BASE}/cublas/lib:\
${_NV_BASE}/nccl/lib:\
${_NV_BASE}/cusparse/lib:\
${_NV_BASE}/cufft/lib:\
${_NV_BASE}/curand/lib:\
${_NV_BASE}/cusolver/lib:\
${_NV_BASE}/nvjitlink/lib:\
${_NV_BASE}/nvtx/lib:\
${LD_LIBRARY_PATH:-}"

unset _NV_BASE

# Default to the only T4 device on this host.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Convenience alias: prefer python3.11 as the project interpreter.
export APOLARITY_PYTHON="${APOLARITY_PYTHON:-/usr/bin/python3.11}"

# Make `python` resolve to python3.11 inside scripts that invoke `python`.
if ! command -v python >/dev/null 2>&1; then
  alias python="$APOLARITY_PYTHON"
fi
