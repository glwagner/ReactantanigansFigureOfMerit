## Figure-of-merit driver for the Reactantanigans NESAP project.
##
## Runs the compressible split-explicit *supercell* (same physics as
## Breeze.jl/examples/scaling_supercell.jl) as a WEAK-SCALING test: the per-GPU
## grid and the grid spacing Δx are held fixed, so the total domain grows with
## the number of ranks. This is the right knob for pushing to 50 m resolution
## across 16 / 32 / 64 GPUs without shrinking the per-GPU work.
##
## From a short timed step loop we report two numbers:
##
##   1. wall_per_10min  — seconds of wall time to advance 10 minutes of
##                        simulated time  =  (s/step) × (600 s / Δt).
##
##   2. FOM             — back-of-the-envelope cost of computing the gradient of
##                        a solution metric w.r.t. the initial condition by
##                        brute-force forward differentiation:
##
##                            FOM = 2 · Nx · Ny · Nz · cost_forward_run
##
##                        where cost_forward_run is the wall time of ONE forward
##                        run over the metric window (--metric-minutes), and
##                        2·Nx·Ny·Nz is the number of tangent (JVP) evaluations
##                        (one per initial-condition degree of freedom, ×2 for
##                        the forward-mode overhead). See the printed caveat on
##                        prognostic-field count.
##
## CLI flags (all optional, env-friendly):
##   --px <Int>            partition along x (default 1)
##   --py <Int>            partition along y (default 1)
##   --dx <Float>          horizontal grid spacing in metres (default 50)
##   --nx-per-gpu <Int>    Nx per rank (default 256)
##   --ny-per-gpu <Int>    Ny per rank (default 256)
##   --nz <Int>            vertical resolution (default 128)
##   --lz <Float>          domain height in metres (default 20000)
##   --dt <Float>          outer time step in seconds (default 0.5)
##   --substeps <Int>      acoustic substeps per outer step (default 12)
##   --metric-minutes <Float> forward-run length for the FOM (default 10)
##   --warmup-steps <Int>  (default 5)
##   --bench-steps <Int>   (default 30)
##   --float-type Float32|Float64 (default Float32)
##   --no-nccl             force MPI Distributed (default uses NCCL)
##
## rank 0 writes one machine-readable line:
##   FOM_CSV,Ngpus,Px,Py,dx,Nx,Ny,Nz,Nx_local,Ny_local,Nz_local,backend,\
##       dt,substeps,ms_per_step,wall_per_10min_s,metric_minutes,\
##       cost_forward_run_s,FOM_gpu_seconds

using MPI
MPI.Init()

using Breeze
using Breeze: CompressibleDynamics
using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization
using Oceananigans: Oceananigans
using Oceananigans.Units

using CUDA
using NCCL
using Printf

## NCCLDistributed is a DistributedComputations function stub, given methods by
## OceananigansNCCLExt (triggered by `using NCCL` + `using CUDA` above).
using Oceananigans.DistributedComputations: NCCLDistributed

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : ARGS[i+1]
end

Px             = parse(Int, argval("--px", "1"))
Py             = parse(Int, argval("--py", "1"))
Δx             = parse(Float64, argval("--dx", "50"))
Nx_per_gpu     = parse(Int, argval("--nx-per-gpu", "256"))
Ny_per_gpu     = parse(Int, argval("--ny-per-gpu", "256"))
Nz             = parse(Int, argval("--nz", "128"))
Lz             = parse(Float64, argval("--lz", "20000"))
Δt             = parse(Float64, argval("--dt", "0.5"))
substeps       = parse(Int, argval("--substeps", "12"))
metric_minutes = parse(Float64, argval("--metric-minutes", "10"))
Nwarmup        = parse(Int, argval("--warmup-steps", "5"))
Nbench         = parse(Int, argval("--bench-steps", "30"))
FT             = Dict("Float32" => Float32, "Float64" => Float64)[argval("--float-type", "Float32")]
use_nccl       = !("--no-nccl" in ARGS)

Oceananigans.defaults.FloatType = FT

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)

Px * Py == Ngpus || error("Px*Py = $(Px*Py) must equal Ngpus = $Ngpus")

## Weak scaling: fixed per-GPU work and fixed Δx, so the total grid and the
## total domain both grow with the rank partition.
Nx = Nx_per_gpu * Px
Ny = Ny_per_gpu * Py
Lx = Δx * Nx
Ly = Δx * Ny

arch = if Ngpus == 1
    GPU()
elseif use_nccl
    NCCLDistributed(GPU(); partition=Partition(Px, Py))
else
    Distributed(GPU(); partition=Partition(Px, Py))
end

if rank == 0
    backend = Ngpus == 1 ? "serial" : (use_nccl ? "NCCL" : "MPI")
    @info "FOM supercell (weak scaling)" Ngpus Px Py FT Δx Nx Ny Nz Lx Ly Lz Δt substeps backend
end

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

time_discretization = SplitExplicitTimeDiscretization(FT; substeps)
dynamics = CompressibleDynamics(time_discretization;
                                surface_pressure = 100000,
                                reference_potential_temperature = 300)

## Supercell initial condition (Klemp et al. 2015), identical to scaling_supercell.jl.
θ₀, θᵖ, zᵖ, Tᵖ = 300, 343, 12000, 213
zˢ, uˢ, uᶜ = 5kilometers, 30, 15

g   = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z <= zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) <= 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

Δθ, rᵇʰ, rᵇᵛ, zᵇ = 3, 10kilometers, 1500, 1500
xᵇ, yᵇ = Lx / 2, Ly / 2

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

advection = WENO(order=5)

## Microphysics off — isolates the dynamics cost (matches scaling_supercell.jl).
model = AtmosphereModel(grid; dynamics, advection, thermodynamic_constants=constants, tracers=())

set!(model, θ=θᵢ, u=uᵢ)

function run_steps!(model, Nt, Δt)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

if rank == 0
    Nx_local, Ny_local, Nz_local = size(grid, 1), size(grid, 2), size(grid, 3)
    @info @sprintf("local grid: %d × %d × %d", Nx_local, Ny_local, Nz_local)
end

MPI.Barrier(MPI.COMM_WORLD)

## Warmup (JIT)
warmup_elapsed = @elapsed run_steps!(model, Nwarmup, Δt)
MPI.Barrier(MPI.COMM_WORLD)

## Two timed trials so run-to-run variation is visible.
trials = Float64[]
for t in 1:2
    MPI.Barrier(MPI.COMM_WORLD)
    el = @elapsed run_steps!(model, Nbench, Δt)
    MPI.Barrier(MPI.COMM_WORLD)
    push!(trials, el)
end

if rank == 0
    best        = minimum(trials)
    s_per_step  = best / Nbench
    ms_per_step = 1000 * s_per_step

    ## Wall time to advance a fixed span of simulated time.
    steps_per_10min = 600 / Δt
    wall_per_10min  = s_per_step * steps_per_10min             # seconds of wall / 10 min sim

    ## One forward run over the metric window, then the brute-force forward-diff FOM.
    cost_forward_run = wall_per_10min * (metric_minutes / 10)  # seconds of wall
    Ncells           = Nx * Ny * Nz
    FOM              = 2 * Ncells * cost_forward_run           # GPU-seconds on this rank count

    backend = Ngpus == 1 ? "serial" : (use_nccl ? "NCCL" : "MPI")
    Nx_local, Ny_local, Nz_local = size(grid, 1), size(grid, 2), size(grid, 3)

    ## Interpretation of the FOM: FOM is the serial wall time (on ONE run at a
    ## time using this GPU count). Multiplying by Ngpus gives total GPU-hours.
    fom_gpu_hours = FOM * Ngpus / 3600
    fom_years     = FOM / (365.25 * 86400)

    @info @sprintf("Warmup: %.3f s  Trial1: %.3f s  Trial2: %.3f s  (best %.3f s, %.3f ms/step)",
                   warmup_elapsed, trials[1], trials[2], best, ms_per_step)

    println("="^70)
    println(" Reactantanigans figure of merit — supercell, weak scaling")
    println("="^70)
    @printf(" GPUs:                 %d  (%d × %d, %s)\n", Ngpus, Px, Py, backend)
    @printf(" resolution:           Δx = %.0f m,  Nz = %d over %.0f m (Δz ≈ %.0f m)\n",
            Δx, Nz, Lz, Lz / Nz)
    @printf(" total grid:           %d × %d × %d  (%.3g cells)\n", Nx, Ny, Nz, Float64(Ncells))
    @printf(" per-GPU grid:         %d × %d × %d\n", Nx_local, Ny_local, Nz_local)
    @printf(" outer Δt / substeps:  %.3f s / %d\n", Δt, substeps)
    @printf(" ms per step:          %.3f\n", ms_per_step)
    @printf(" WALL PER 10 MIN SIM:  %.2f s   (%.0f steps / 10 min)\n", wall_per_10min, steps_per_10min)
    println("-"^70)
    @printf(" metric window:        %.0f min\n", metric_minutes)
    @printf(" cost_forward_run:     %.2f s wall\n", cost_forward_run)
    @printf(" FOM = 2·Nx·Ny·Nz·cost_forward_run:\n")
    @printf("     = %.3g GPU-seconds (serial wall on %d GPUs)\n", FOM, Ngpus)
    @printf("     = %.3g GPU-hours\n", fom_gpu_hours)
    @printf("     = %.3g years of wall time (serial on %d GPUs)\n", fom_years, Ngpus)
    println("-"^70)
    @printf(" NOTE: Nx·Ny·Nz counts CELLS. The true initial-condition DOF count is\n")
    @printf("       ~5× larger (ρ, ρu, ρv, ρw, ρθ), so multiply the FOM by ~5 for the\n")
    @printf("       full state gradient. This cost is exactly what motivates adjoint\n")
    @printf("       (reverse-mode) differentiation over brute-force forward diff.\n")
    println("="^70)

    println("FOM_CSV,$Ngpus,$Px,$Py,$Δx,$Nx,$Ny,$Nz,$Nx_local,$Ny_local,$Nz_local,$backend,$Δt,$substeps,$ms_per_step,$wall_per_10min,$metric_minutes,$cost_forward_run,$FOM")
end
