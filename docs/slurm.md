# Running the `mpi` backend under Slurm

**Status: deployment guidance, not measured on a Slurm cluster here** (this dev
box has no `srun`/`sinfo`; the MPI is MPICH 4.3.2 `ch4:ofi` with Hydra, and
`mpiexec -n N` works).  Everything below follows from how Slurm launches MPI
steps; the `mpi` backend's own behaviour is measured in
`docs/mpi_backend.md`.

## 1. How Slurm runs MPI — and how it silently does not

Slurm launches tasks; it does **not** speak MPI.  It provides a PMI (PMI-1/PMI-2)
or PMIx server for each step, and the MPI library initialises by talking to it.
A launcher (`mpiexec`) is then unnecessary.  If the MPI library was not built
against that server, `srun` still starts N processes — but each one believes
`COMM_WORLD` has size 1, with no error.  For the `mpi` backend that means every
rank computes the *whole* sample and the `allreduce` never happens.

Check the build before writing the job:

| MPI | check | launch under Slurm |
|---|---|---|
| OpenMPI built with PMIx | `ompi_info --param pmix all` | `srun --mpi=pmix …` |
| OpenMPI built with PMI-2 | `ompi_info --param pmi2 all` | `srun --mpi=pmi2 …` |
| MPICH built with `--with-pmi=slurm`/`pmi2` | `mpichversion \| grep -i pmi` | `srun …` |
| MPICH/Hydra standalone (conda, **this box**: `ch4:ofi`, no `--with-pmi`) | no pmi line | **`mpiexec` inside the allocation** (below) |
| Intel MPI | `I_MPI_PMI_LIBRARY` | `srun …` |

**Never nest launchers** (`srun mpiexec -n …`): two process launchers in one
step oversubscribe and produce failures far from the cause.  Choose one row.

## 2. Recommended shape: one task per GPU, MPI across nodes

```bash
#!/bin/bash
#SBATCH --job-name=tabpwa-fit
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-task=1            # Slurm gives each task its own device
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j/slurm-%t.out   # one log PER TASK (see §5)
#SBATCH --signal=B:INT@1800          # checkpoint before the wall (see §6)

set -euo pipefail
# The MPI that mpi4py was built against must be the one Slurm launches.
# Prefer the cluster module (module load openmpi/…), not a conda MPICH,
# unless the conda env's MPI is itself Slurm-aware.

# A) MPI built for Slurm PMI/PMIx:
srun --mpi=pmix python run_fit.py --config config_pwa.yml --backend mpi --fit
```

For a standalone-Hydra MPI (or when `srun` gives size-1 jobs), launch **inside**
the allocation from a hostfile and do not use `srun` for the step:

```bash
HOSTS="$SLURM_TMPDIR/hosts"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOSTS"
mpiexec -f "$HOSTS" -ppn "$SLURM_NTASKS_PER_NODE" -n "$SLURM_NTASKS" \
        python run_fit.py --config config_pwa.yml --backend mpi --fit
```

Both forms give every rank the same rank/size (`mpi_backend` reads
`COMM_WORLD`), the same sample on the shared filesystem, and rank 0 as the only
writer.

## 3. GPU assignment — what our binding does with Slurm's environment

Slurm exports `CUDA_VISIBLE_DEVICES` per task.  `mpi_backend.device_index`
handles the three cases:

| `CUDA_VISIBLE_DEVICES` | meaning | backend |
|---|---|---|
| *(unset)* | no launcher mapping | `local_rank` → device (needs `nvidia-smi`); a rank past the count raises |
| one entry (`--gpus-per-task=1`) | this task owns that device | **left alone** (device 0 in the task's view) |
| `0,1,2,3` (`--gres=gpu:4`, node-level) | every task sees all four | **indexed by the node-local rank**, so rank *r* gets entry *r*; a rank past the list raises |
| `""` | hide every device | left alone |

So `--gpus-per-task=1` is the simplest: Slurm owns the mapping and the backend
never touches it.  `--gres=gpu:N` (node-level) is also correct, but only because
the backend now indexes the list — before, every rank would have used the first
visible device and quietly shared one GPU.

## 4. Data and memory

* Keep the sample on the shared filesystem and **generate it once** (a serial
  process or a pre-existing file).  `Fitter.generate_phsp`/`generate_toy` drive
  the backend per batch, so on an MPI fitter they are a collective — correct,
  but N× duplicated work (`docs/mpi_backend_design.md` §6).
* Every rank currently loads the **full** buffer before slicing (`N×` per node,
  the tracked follow-up): a 4-GPU node with a 600 MB sample holds ≈ 2.4 GB of
  duplicates.  For multi-node runs this is the number to watch; the rank-aware
  loader removes it.
* The per-iteration exchange is `Q` + gradients (KB); the reduction cost is
  irrelevant next to the sample load and the data pass.

## 5. Output and files

* `run_fit.py` guards its **writes** (save, save-init, plot) to rank 0, so a
  shared filesystem is safe.
* It does **not** silence the other ranks' prints.  Slurm merges a step's
  stdout, so use `--output=…-%t.out` (per task) — as above — or accept N
  interleaved logs.
* Results (`*_fit_results.json`, `error_matrix.npy`) are written by rank 0 to
  the shared filesystem.

## 6. Signals and checkpoints

Slurm's `scancel` sends **SIGTERM** by default; the fitter's interrupt
checkpoint (`_last_xk` → `checkpoint.json`) is wired to **SIGINT**
(`KeyboardInterrupt`).  A plain `scancel` therefore loses the run.

* `scancel --signal=INT <jobid>` — send SIGINT instead;
* `#SBATCH --signal=B:INT@1800` — Slurm sends SIGINT 30 min before the wall
  limit (the `B:` prefix targets the batch shell);
* before the deadline, `--time` should exceed the fit's expected wall time; the
  fit is sequential over BFGS iterations, so the per-iteration count is the
  lever (`--maxiter`).

## 7. Troubleshooting

| symptom | likely cause |
|---|---|
| every rank reports `size=1` (or the fit runs N times on the whole sample) | the MPI is not PMI/PMIx-enabled for `srun`; use `--mpi=pmix`/`pmi2`, or the `mpiexec -f hostfile` form |
| job hangs at start / in a collective | one rank died before the collective — the backend now `Abort`s (job fails fast); if it still hangs, check that **every** rank runs the same script (a rank-guarded `if rank == 0: fit` cannot work) |
| all ranks on one GPU | node-level `--gres` with a stale backend (fixed: the list is indexed by local rank), or no `--gpus-per-task` and a launcher-set `CUDA_VISIBLE_DEVICES` |
| `rank … has no device` | more ranks than visible devices; allocate one GPU per task or launch fewer ranks |
| N copies of the log | merged stdout — use per-task `--output` |
| `mpi4py` import error / MPI mismatch | the env's MPI differs from the one Slurm launched; build `mpi4py` against the cluster module (`pip install mpi4py` *after* `module load`) |
| sample load is N× slower than serial | expected until rank-aware loading lands |

## 8. When *not* to use MPI under Slurm

* One node: `shard` does the same reduction with `multiprocessing` and no MPI
  dependency (its cost is chunk pickling, `docs/mpi_backend_design.md` §1).
* The scheduler already gives you independent jobs: a parameter scan is N
  separate fits, not one MPI job.