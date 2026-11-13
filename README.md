# flink-jitsql

**Flink SQL operators on a GPU, through TornadoVM.**

Runs bounded batch Table API/SQL operators on an NVIDIA GPU. The query author writes
ordinary Flink SQL and nothing else — no API, no annotation, no hint, no per-query
option — and the planner decides what is worth moving.

```sql
SELECT id, lat, lon, 6371.0 * 2 * ASIN(SQRT(...)) AS km FROM Points WHERE ...
```

That query runs on the device. Nothing in it says so, and that is the claim: the
answer does not depend on where it ran.

## Why

Flink discovers an accelerator through `ServiceLoader` and knows nothing about the one
it finds — not the framework, not the device API, not the vendor. This repository is
the other side of that interface. It holds everything Flink must not: kernel
generation, buffer layouts, staging widths, and every constant calibrated against a
particular card.

### Why it is not part of Flink

It was, and the build gave the coupling away: Flink's own reactor depended on a vendor
runtime — a CUDA toolchain, a driver version floor, JVM module-path requirements — and
the planner emitted `uk.ac.manchester.tornado.api.types.arrays.DoubleArray` into
generated source. An extension point cannot be proposed on those terms, because the
vendor *is* the interface.

TornadoVM's artifacts were also unpublished then, and that is the part people noticed.
They are on Maven Central now and it changes nothing: a checkout of Flink still must
not need a GPU toolchain to build. Flink ships the work description; this ships the
rate and the kernel.

## What you need

| | |
|---|---|
| GPU | NVIDIA, compute capability 7.0+, 8 GiB or more |
| Driver | 525 or newer, for CUDA 12 |
| JDK | **exactly 21** — TornadoVM's off-heap arrays are `java.lang.foreign`, preview on 21 and final on 22, and Flink 2.3 has no JDK 22+ profile |
| Maven | 3.9 or newer |
| TornadoVM | **8.1.0**, from Maven Central to compile. To reach a device: `sdk install tornadovm 8.1.0-jdk21-cuda`. The released SDK is enough, the cuDF path included — 8.1.0 ships `lib/libtornado-cudf.so` built, where earlier releases did not |
| RAPIDS libcudf | for the operators cuDF serves. `source scripts/cudf-env.sh` installs it from NVIDIA's wheels |
| Apache Flink | the accelerator SPI, which is in no release — branch `gpu-offload` of [`beestream-ai/flink`](https://github.com/beestream-ai/flink) |

Only `mvn test` is needed to run the 124 unit tests. Everything else is needed to
reach a GPU.

## Getting it running

```bash
# the Flink SPI, which is not released
git clone --branch gpu-offload https://github.com/beestream-ai/flink && cd flink
./mvnw install -DskipTests -Dfast -pl flink-table/flink-table-common,flink-table/flink-table-runtime -am
cd ..

git clone https://github.com/beestream-ai/flink-jitsql && cd flink-jitsql
export JAVA_HOME=/path/to/jdk-21
export TORNADOVM_HOME=$HOME/.sdkman/candidates/tornadovm/8.1.0-jdk21-cuda
source scripts/cudf-env.sh     # RAPIDS libcudf, six directories of it
scripts/preflight.sh           # says what is wrong before a run does
mvn install
```

Point `TORNADOVM_HOME` at the **versioned** directory, never at
`~/.sdkman/candidates/tornadovm/current`: `current` follows whatever was installed
last, and here it resolved to a 2.2.0 OpenCL SDK while everything was built against
8.1.0 CUDA. `preflight.sh` catches exactly that.

Every path is overridable; `scripts/env.sh` lists them and derives the rest from the
repository root, so a clone anywhere works without editing anything.

## Did it work

```bash
scripts/run-sql-demos.sh check      # every operator shape, both arms, compared row by row
```

must end:

```
CHECKS: 10, FAILURES: 0
```

Each case says whether **the planner** put the subtree on the device, and when it did
not, the planner's own reason — `a sort key of type DOUBLE NOT NULL is not ordered on
device`, and so on. A case that agrees while the plan held nothing offloaded has checked
that the CPU equals the CPU; it says `SAME (cpu)` rather than counting as a device result.

`(device)` is the plan's verdict, not a runtime trace: the planner does not know whether
this TaskManager has RAPIDS, so a cuDF-served operator is offloaded in the plan and
declined at run time if the shim is missing. To watch the device actually execute one:

```bash
scripts/run-sql-demos.sh cudf-sort --printBytecodes
```

which prints `LAUNCH task - sort.order[sortedOrder] on [NVIDIA CUDA]`.

If everything says `cpu`, run the smoke test before suspecting this code:

```bash
scripts/run-sql-demos.sh smoke      # one kernel, no Flink. Ends: SMOKE: OK
```

It separates *TornadoVM is not working here* from *this project is not reaching it*.
The first is far more common and is not a defect in anything here.

## Two examples to read first

| example | what runs on the device | how you see that it did |
|---|---|---|
| `HaversineSQLExample` | a **CUDA kernel generated from the query**. Flink lowers the projection to the accelerator IR, the provider compiles that IR at job start, and the kernel exists only for this query. | `--printKernel` prints the generated `evaluate`, with `sin`, `pow` and `asin` in it |
| `CudfSortSQLExample` | an **`ORDER BY` served by a library**. No generated kernel at all: the provider recognises the operator and hands it to `cudf::stable_sorted_order` through TornadoVM's `tornado-cudf` binding. | `--printBytecodes` prints `LAUNCH task - sort.order[sortedOrder]` on the CUDA device |

```bash
flink-jitsql-plugin/scripts/run-sql-demos.sh haversine --printKernel
flink-jitsql-plugin/scripts/run-sql-demos.sh cudf-sort --printBytecodes
```

Both run in one JVM against an in-process MiniCluster, so there is no cluster to
start. Neither says anywhere in its output where the query ran, and that is the
claim: the answer does not depend on it. The evidence is asked of TornadoVM
instead, which is why `--printKernel` prints something no file in this repository
contains.

`--printKernel` prints **nothing** for the cuDF example, and that absence is the
point rather than a gap: there is no generated kernel, because the operator was
served by a library.

The script assembles the four things that are properties of the deployment rather
than of the query — TornadoVM's JVM arguments, the flags that make device
arithmetic agree with the CPU's, RAPIDS on `LD_LIBRARY_PATH`, and the provider jar
— and validates them before starting a JVM. `VERBOSE=1` turns on the accelerator's
own logging, which is what to do if no kernel and no `LAUNCH` appear. See
`DEPLOYMENT.md`.

**Neither example measures anything.** They demonstrate that the mechanism works
and let you see it working. For a number, with an environment record, repetitions
and a CPU arm behind it, see
[`flink-jitsql-benchmarking`](https://github.com/beestream-ai/flink-jitsql-benchmarking).

## How to see why a query did not offload

Two surfaces, and the split is deliberate — a planner's reasons are about the query,
a TaskManager's are about the machine it turned out to have.

| | |
|---|---|
| `EXPLAIN` | what the **planner** decided, per subtree: `GPU  subtree 0: ...` or `CPU  ...` with the reason under it. Properties of the query — an operator with no accelerator IR, a type that is not ordered on device, a projection whose arithmetic may round differently |
| metrics | what the **TaskManager** decided, per subtask, as `AcceleratorDeclineReason`: `ACCELERATED`, `NO_PROVIDER`, and the rest. Properties of the machine — no provider on the classpath, no device, a driver that did not load |

`NO_PROVIDER` is the ordinary outcome on most machines and is not an error. It is also
the one that most often surprises someone who read an `EXPLAIN` saying otherwise.

## What it handles

Bounded batch `Calc` (projection and filter), `GROUP BY`, joins, sorts and over-aggregates,
over fixed-width numeric columns. Generated CUDA kernels for the arithmetic; cuDF through
TornadoVM's binding for the relational primitives a kernel cannot express — a grouped
aggregate needs cross-row cooperation that no `@Parallel` loop states.

## What it does not do

- **Streaming.** Explicit non-goal for v1: output is deferred by a full batch and staging
  state does not survive a checkpoint barrier.
- **Nullable expression inputs.** `NOT NULL` in the DDL is the one thing a query author
  has to write, and the one sanctioned exception to writing nothing.
- **Strings, decimals, structs** in a kernel, and any operator shape the planner declines.

All of these are refused **while planning**, so the query keeps the path it already had.
Planning-time refusal is not executor-time fallback: once a plan holds a kernel, a failure
during execution fails the task.

## Status

Research code. One machine, one card, bounded batch, Flink `local` and a single-node
cluster. No multi-GPU and no production claim. The SPI it implements is in no Flink
release and lives on a fork; until a FLIP lands, that fork has to be built first.

Numbers, with the environment they were measured in, are in
[`flink-jitsql-benchmarking`](https://github.com/beestream-ai/flink-jitsql-benchmarking).
Deployment, the build order and the couplings that fail at run time rather than at build
time are in [DEPLOYMENT.md](DEPLOYMENT.md).

## License

flink-jitsql is released under the [Apache License, Version 2.0](LICENSE). Copyright (c)
2026 Beestream Ltd. Every source file carries the matching header.

Nothing third-party is included here; what the build and the examples fetch from elsewhere
is listed in [NOTICE](NOTICE). In particular, the Flink SPI this implements stays under the
Apache Software Foundation's licence and keeps its file headers, and no file here ships in
Flink's package namespace.
