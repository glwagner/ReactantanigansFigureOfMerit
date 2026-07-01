#!/bin/bash
#
# Reactantanigans figure-of-merit weak-scaling sweep on Perlmutter.
#
# Fixed per-GPU work and fixed Δx (default 50 m); the domain and total grid grow
# with the GPU count. Sweeps GPU counts in one allocation, one srun step per
# count, and prints wall-time-per-10-min and the forward-diff FOM for each.
#
# Perlmutter has 4 A100 GPUs per node, so 16/32/64 GPUs = 4/8/16 nodes.
#
# DEBUG (default): the debug QOS caps at 8 nodes / 30 min. The default is the
# 8-node (32-GPU) FOM; add 16 to also get the 4-node point for weak scaling:
#
#   sbatch figure_of_merit.sh                       # GPUS=32 on debug, 8 nodes
#   GPUS="16 32" sbatch figure_of_merit.sh          # both points, one allocation
#
# 64 GPUs (16 nodes) needs the regular QOS; override the in-file directives on
# the command line (sbatch flags win over #SBATCH):
#
#   GPUS=64 sbatch --qos=regular --nodes=16 --time=00:20:00 figure_of_merit.sh
#
#SBATCH --job-name=fom-sweep
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=fom-sweep-%j.out
#SBATCH --error=fom-sweep-%j.err

set -u

REPO_DIR="${REPO_DIR:-/global/u1/g/glwagner/ReactantanigansFigureOfMerit}"
PROJECT_DIR="${PROJECT_DIR:-/global/u1/g/glwagner/Breeze.jl}"   # reuse Breeze's tuned examples env
cd "${REPO_DIR}"

module load julia/1.12.1
JULIA="${JULIA:-julia}"

## See Breeze's BREEZE_ON_PERLMUTTER.md / NCCL_PERLMUTTER.md for the why on each.
JULIA_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v "${JULIA}")")")")
export LD_LIBRARY_PATH="${JULIA_PREFIX}/lib/julia:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="${CRAY_MPICH_ROOTDIR:-/opt/cray/pe/mpich/9.0.1}/gtl/lib/libmpi_gtl_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export JULIA_CUDA_MEMORY_POOL=none
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"

## NCCL AWS-OFI plugin for CXI/Slingshot RDMA (otherwise NCCL falls back to slow
## TCP sockets inter-node). Set PLUGIN=2.18.3 to enable (matches weak_scaling_sweep.sh).
if [ -n "${PLUGIN:-}" ]; then
    export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/${PLUGIN}/lib:${LD_LIBRARY_PATH}"
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

## Fixed per-GPU problem (weak scaling) and physics parameters.
DX="${DX:-50}"
NX_PER_GPU="${NX_PER_GPU:-256}"
NY_PER_GPU="${NY_PER_GPU:-256}"
NZ="${NZ:-128}"
LZ="${LZ:-20000}"
DT="${DT:-0.5}"
SUBSTEPS="${SUBSTEPS:-12}"
METRIC_MINUTES="${METRIC_MINUTES:-10}"
WARMUP="${WARMUP:-5}"
BENCH="${BENCH:-30}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
NCCL="${NCCL:-1}"
BACKENDFLAG=""
[ "${NCCL}" = "0" ] && BACKENDFLAG="--no-nccl"

# 2D rank partition (Px Py) for each GPU count; kept as square as possible.
partition_for() {
    case "$1" in
        1)  echo "1 1" ;;
        2)  echo "2 1" ;;
        4)  echo "2 2" ;;
        8)  echo "4 2" ;;
        16) echo "4 4" ;;
        32) echo "8 4" ;;
        64) echo "8 8" ;;
        *)  echo "" ;;
    esac
}

echo "=========================================="
echo "Reactantanigans FOM sweep -- Perlmutter"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-<none>}"
echo "Node(s):       ${SLURM_NODELIST:-<none>}"
echo "Per-GPU grid:  ${NX_PER_GPU} × ${NY_PER_GPU} × ${NZ}"
echo "Δx:            ${DX} m     Δt: ${DT} s     substeps: ${SUBSTEPS}"
echo "Metric window: ${METRIC_MINUTES} min"
echo "Backend:       $([ "${NCCL}" = 0 ] && echo MPI || echo NCCL)     Float: ${FLOAT_TYPE}"
echo "julia:         $(${JULIA} --version)"
echo "=========================================="

## Preflight: warm the depot with a SINGLE task first, precompiling exactly the
## modules the run loads. Launching the parallel srun against a cold depot makes
## every rank precompile at once and contend on one pidfile in shared scratch —
## a "precompile storm" that can eat an entire allocation. We deliberately do NOT
## `Pkg.precompile()` the whole project: that pulls in BreezeCloudMicrophysicsExt,
## which currently fails to precompile (UndefVarError: `Open`) and is unused here
## (dry dynamics only — `using Breeze` alone does not load CloudMicrophysics).
echo ">>> Preflight: serial warm of the exact module tree (avoids precompile storm)"
srun --ntasks=1 --gpus=1 --gpu-bind=none \
    "${JULIA}" --project="${PROJECT_DIR}/examples" -e '
        using MPI, Oceananigans, CUDA, NCCL
        using Breeze
        using Breeze: CompressibleDynamics
        using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization
        println("PREFLIGHT_OK")' \
    || echo "preflight warm returned rc=$?"

## With the depot warm, forbid the parallel ranks from auto-precompiling: they
## must load from cache. This makes any remaining gap fail fast instead of
## triggering a 32-way precompile storm.
export JULIA_PKG_PRECOMPILE_AUTO=0

for N in ${GPUS:-32}; do
    read -r PX PY <<< "$(partition_for "${N}")"
    if [ -z "${PX}" ]; then
        echo ">>> GPUs=${N}: no partition defined, skipping"
        continue
    fi
    echo ""
    echo ">>>>>> GPUs = ${N}  (${PX} × ${PY}) <<<<<<"
    srun --ntasks="${N}" --ntasks-per-node=4 --gpus="${N}" --gpu-bind=none \
        "${JULIA}" --project="${PROJECT_DIR}/examples" figure_of_merit.jl \
            --px "${PX}" --py "${PY}" \
            --dx "${DX}" --nx-per-gpu "${NX_PER_GPU}" --ny-per-gpu "${NY_PER_GPU}" \
            --nz "${NZ}" --lz "${LZ}" --dt "${DT}" --substeps "${SUBSTEPS}" \
            --metric-minutes "${METRIC_MINUTES}" \
            --warmup-steps "${WARMUP}" --bench-steps "${BENCH}" \
            --float-type "${FLOAT_TYPE}" ${BACKENDFLAG} \
        || echo "GPUs=${N} FAILED (rc=$?)"
done

echo ""
echo "=========================================="
echo "Sweep done. Summary (one line per GPU count):"
echo "=========================================="
echo "Ngpus,Px,Py,dx,Nx,Ny,Nz,backend,dt,substeps,ms_per_step,wall_per_10min_s,metric_min,cost_forward_s,FOM_gpu_seconds"
grep -h '^FOM_CSV' "fom-sweep-${SLURM_JOB_ID}.out" 2>/dev/null | sed 's/^FOM_CSV,//'
