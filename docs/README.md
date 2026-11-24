# Documentation

Two kinds, kept apart on purpose:

* **design** — what should exist and why.  Normative, and allowed to describe
  something that is not built yet.  It carries no measurements of the tree.
* **implementation** — what the tree actually does: files, API, measured
  behaviour, gates, limits.  It records deviations from the design.

A design and its implementation are SEPARATE files.  If they disagree, the
design is the intent and the implementation file is the bug.  Write a change in
the design first; then update the implementation file with what was built.

| file | kind | status |
|---|---|---|
| [`mpi_backend_design.md`](mpi_backend_design.md) | design | implemented (v1) |
| [`mpi_backend.md`](mpi_backend.md) | implementation | in the tree; measurements + gates |
| [`slurm.md`](slurm.md) | implementation / deployment | guidance for Slurm jobs |
| [`mpi_service_design.md`](mpi_service_design.md) | design | implemented (v1) — driver + subprocess MPI workers |
| [`mpi_service.md`](mpi_service.md) | implementation | in the tree; measurements + gates |
| [`remote_protocol_design.md`](remote_protocol_design.md) | design (draft standard) | **not implemented** (prototype stashed; open questions in §14) |
| [`likelihood_layer.md`](likelihood_layer.md) | design | **not implemented** |
| [`path_map.md`](path_map.md) | implementation / usage | in the tree |

The backend/kernel architecture lives in `AGENTS.md`; this directory holds the
design and deployment documents that would make that file unwieldy.
