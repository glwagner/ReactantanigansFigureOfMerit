# ReactantanigansFigureOfMerit

Figure of merit for the Reactantanigans NESAP project.

## The problem

We want to compute the gradient of a scalar metric of an atmospheric simulation
(e.g. accumulated precipitation, peak updraft velocity, storm track) with respect
to the **initial condition**. This is the core kernel of variational data
assimilation, sensitivity analysis, and gradient-based optimization.

The initial condition is the full model state on the grid: `Nx · Ny · Nz` cells,
each carrying several prognostic fields. To motivate why the project needs
efficient adjoint (reverse-mode) differentiation, we quantify the cost of the
**naive alternative** — computing that same gradient by brute-force *forward*
differentiation, one perturbed initial-condition degree of freedom at a time.

The FOM is that cost:

```
FOM = 2 · Nx · Ny · Nz · cost_forward_run
```

- `Nx · Ny · Nz` — number of initial-condition degrees of freedom (grid cells);
  forward mode needs one tangent (JVP) evaluation per DOF to fill in the gradient.
- `2 ·` — forward-mode AD overhead (a JVP costs roughly twice a primal run).
- `cost_forward_run` — wall time of one forward run over the metric window.

This number is deliberately enormous — it is what motivates adjoint
(reverse-mode) differentiation, whose cost is ~a few forward runs *regardless* of
the number of inputs, over brute-force forward diff whose cost scales with the
number of inputs.

## Logic for producing the final FOM figure

The only quantity that must be *measured* is `cost_forward_run`; the rest of the
FOM is arithmetic on the grid size. The pipeline is:

1. **Measure `ms/step`** at the production configuration — fully compressible,
   split-explicit, Δx = 50 m — by timing a short step loop on GPU
   (`figure_of_merit.jl`, native `time_step!`, best of two trials after warmup).
2. **Convert to wall-per-10-min:** `wall_per_10min = (ms/step / 1000) · (600 / Δt)`.
   The number of steps in a fixed simulated span is `600 / Δt`.
3. **Scale to the metric window:** `cost_forward_run = wall_per_10min · (metric_minutes / 10)`.
   This is the wall time of the single forward run whose gradient we want.
4. **Apply the FOM formula:** `FOM = 2 · Nx · Ny · Nz · cost_forward_run`, then
   express it in GPU-hours (`× Ngpus / 3600`) and wall-clock years for context.
5. **Establish weak scaling.** Because the per-GPU work and Δx are fixed while the
   domain grows, `cost_forward_run` should stay ~flat across 16 → 32 → 64 GPUs
   (any rise is communication overhead). Flat cost is what makes the FOM at the
   largest configuration a fair estimate of the full-scale problem rather than an
   artifact of a small test.

**The final FOM figure** is the value at the production configuration: the target
resolution (Δx = 50 m), the largest domain / GPU count we run (64 GPUs weak-scaled),
and the science-relevant metric window (`--metric-minutes`). The weak-scaling sweep
across 16/32/64 GPUs supplies the confidence that `cost_forward_run` is stable, so
the headline number is the cost measured at 64 GPUs fed through steps 2–4 above.
`Nx·Ny·Nz` counts cells; the true state gradient is ~5× larger (see Notes).

## Case

**Supercell** (Klemp et al. 2015), same physics as
`Breeze.jl/examples/scaling_supercell.jl`, with the **fully compressible
formulation** (`CompressibleDynamics`) and **split-explicit** acoustic
substepping (`SplitExplicitTimeDiscretization`); microphysics off to isolate the
dynamics cost. Run as a **weak-scaling** test: the per-GPU grid and the grid
spacing Δx are held fixed, so the total domain grows with the GPU count. This is
the knob for reaching 50 m resolution across 16 / 32 / 64 GPUs without shrinking
the per-GPU work.

## What it reports

Per GPU count, rank 0 prints:

- `ms per step`
- **`WALL PER 10 MIN SIM`** — seconds of wall time per 10 minutes of simulated
  time = `(s/step) × (600 / Δt)`
- `cost_forward_run` over `--metric-minutes`
- the `FOM`, in GPU-seconds, GPU-hours, and wall-clock years

plus a machine-readable `FOM_CSV,...` line.

## Results

### Measured — 1 node, 4× A100 (Perlmutter, 2026-07-01)

Full production per-GPU load (256 × 256 × 128 per GPU, Δx = 50 m, Δt = 0.5 s,
12 acoustic substeps, WENO-5, Float32, GPU-aware MPI):

| quantity | value |
|---|---|
| per-GPU grid | 256 × 256 × 128 |
| total grid | 512 × 512 × 128 (3.36×10⁷ cells) |
| ms / step | **481.8** |
| **wall per 10 min sim** | **578 s** (1200 steps; ≈ 0.96× real time) |
| cost_forward_run (10 min) | 578 s |
| **FOM = 2·Nx·Ny·Nz·cost_forward** | **3.88×10¹⁰ GPU-s** = 4.3×10⁷ GPU-h = 1.2×10³ yr serial on 4 GPUs |

This is the per-GPU weak-scaling baseline: `ms/step` (hence `cost_forward_run`)
is set by the fixed per-GPU work, so it carries directly to higher GPU counts —
only `Nx·Ny·Nz` grows.

### Projected — 8 nodes, 32× A100 (weak scaling)

Under ideal weak scaling (`cost_forward_run` ≈ constant, cells × 8):

| quantity | value |
|---|---|
| total grid | 2048 × 1024 × 128 (2.68×10⁸ cells) |
| cost_forward_run (10 min) | ≈ 578 s (comm overhead not yet measured) |
| **FOM (projected)** | **≈ 3.1×10¹¹ GPU-s** = 2.8×10⁹ GPU-h ≈ 9.8×10³ yr serial on 32 GPUs |

**This projection is not yet a measurement.** See the blocker below.

### ⚠ Inter-node execution is currently blocked

Multi-node runs hang at the first inter-node halo exchange (inside `set!`),
identically for **both** the GPU-aware MPI and NCCL backends:

| config | result |
|---|---|
| 1 node / 4 GPUs (any grid size) | ✅ runs, produces FOM |
| 2 nodes / 8 GPUs | ❌ hangs at first inter-node halo |
| 8 nodes / 32 GPUs (MPI and NCCL) | ❌ hangs → 30-min timeout |

This is **not specific to this driver**: Breeze's own multi-node
`distributed_tropical_cyclone` runs at 8–15 nodes also fail (exit 143, no
stepping). The failure boundary is exactly the node crossing (1 node works,
≥ 2 nodes hang), and no NCCL `WARN` is emitted before the stall — pointing to a
system-level GPU-aware inter-node transport problem (GTL / Slingshot-OFI), not a
bug in the FOM code. Resolving it is a prerequisite for a *measured* 8-node FOM.

## Running on Perlmutter

The debug QOS caps at **8 nodes / 30 min**, which fits 16 and 32 GPUs (4 and 8
nodes) but not 64 (16 nodes). The sweep defaults to debug + `GPUS="16 32"`.

```bash
# 16 + 32 GPUs on debug (default):
sbatch figure_of_merit.sh

# 64 GPUs needs the regular QOS (sbatch flags override the in-file #SBATCH):
GPUS=64 sbatch --qos=regular --nodes=16 --time=00:20:00 figure_of_merit.sh

# Override defaults (Δx = 50 m, 256×256×128 per GPU, Δt = 0.5 s):
GPUS="16 32" DX=50 NX_PER_GPU=256 NY_PER_GPU=256 NZ=128 DT=0.5 \
    METRIC_MINUTES=30 sbatch figure_of_merit.sh
```

The launcher reuses Breeze's tuned `examples` environment
(`--project=$PROJECT_DIR/examples`) for the NCCL / CUDA / MPI stack. Set
`PLUGIN=2.18.3` to enable the NCCL AWS-OFI Slingshot plugin for fast inter-node
collectives (recommended at ≥ 8 GPUs).

### Single run

```bash
srun --ntasks=16 --ntasks-per-node=4 --gpu-bind=none \
    julia --project=$HOME/Breeze.jl/examples figure_of_merit.jl \
        --px 4 --py 4 --dx 50 --nx-per-gpu 256 --ny-per-gpu 256 --nz 128 \
        --dt 0.5 --substeps 12 --metric-minutes 10
```

See the header of `figure_of_merit.jl` for the full flag list.

## Notes

- `Nx·Ny·Nz` counts **cells**. The true initial-condition DOF count is ~5× larger
  (ρ, ρu, ρv, ρw, ρθ); multiply the FOM by ~5 for the full state gradient.
- `Δt = 0.5 s` with 12 acoustic substeps is stable at Δx = 50 m (advective CFL ≈ 0.5,
  acoustic CFL ≈ 0.3). Adjust `--dt` to the physically appropriate step for a
  different resolution — the wall-per-10-min number is linear in `600/Δt`.
