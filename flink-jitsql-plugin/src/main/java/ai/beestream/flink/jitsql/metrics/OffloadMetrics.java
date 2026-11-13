/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.beestream.flink.jitsql.metrics;

/**
 * The gather / copy-in / kernel / copy-out breakdown that is P1's exit criterion.
 *
 * <p>Shaped after Table 3 of "Enabling Transparent Acceleration of Big Data Frameworks Using
 * Heterogeneous Hardware" (PVLDB 15(13):3869-3882), where the equivalent breakdown showed copy-in
 * at 77.4% and kernel at 4.7% of accelerated time. The point of collecting these separately is that
 * they discriminate between different failures: copy-in dominant means device residency is the fix,
 * gather dominant means the input tier is wrong, kernel dominant means the batch is too small.
 *
 * <p>Host times are measured here; device times come from TornadoVM's profiler, so an execution
 * plan must have been built {@code .withProfiler(ProfilerMode.SILENT)} for them to be populated.
 * Values are nanoseconds unless named otherwise.
 */
public final class OffloadMetrics {

    // Volatile throughout, because every one of these is published to Flink's metric reporter as a
    // gauge and read on the reporter's thread while the task thread is still writing. JLS 17.7
    // does not give a non-volatile long an atomic read, so the reporter could see a value that is
    // neither the old one nor the new one -- the high half of one write against the low half of
    // another -- and there would be no way to tell that from a real number. The += below stay
    // correct because the task thread is the only writer; what volatile buys here is a whole read
    // and a visible one, not mutual exclusion.
    private volatile long batches;
    private volatile long rowsIn;
    private volatile long rowsOut;

    private volatile long gatherNanos;
    private volatile long drainNanos;
    private volatile long executeWallNanos;

    private volatile long copyInNanos;
    private volatile long kernelNanos;
    private volatile long copyOutNanos;
    private volatile long compileNanos;

    private volatile long bytesCopyIn;
    private volatile long bytesCopyOut;
    private volatile long onDemandBytesOut;

    private volatile long setupNanos;
    private volatile long setupRows;

    /** Records one batch. {@code result} may be null when the profiler is disabled. */
    public void recordBatch(
            int rows,
            int emitted,
            long gather,
            long executeWall,
            long drain,
            ai.beestream.flink.jitsql.operator.GeneratedKernelEngine.DeviceProfile profile) {
        batches++;
        rowsIn += rows;
        rowsOut += emitted;
        gatherNanos += gather;
        executeWallNanos += executeWall;
        drainNanos += drain;
        addProfile(profile);
    }

    /**
     * Folds in device numbers that did not come from {@link #recordBatch}.
     *
     * <p><b>Not for a second task graph in the same plan.</b> {@code TornadoProfilerResult}
     * delegates to the <em>executor</em>, so every graph of one execution plan answers with the
     * same plan-wide cumulative totals; adding two of them counts everything twice. That was done
     * here and inflated the fused path's reported kernel time and byte counts by a factor of two
     * until a controlled measurement caught it — the give-away was that the "first graph only"
     * figure equalled the whole non-compacted plan's, for a graph that contains no {@code
     * transferToHost} at all.
     */
    public void addProfile(
            ai.beestream.flink.jitsql.operator.GeneratedKernelEngine.DeviceProfile profile) {
        if (profile == null) {
            return;
        }
        copyInNanos += profile.copyInNanos();
        kernelNanos += profile.kernelNanos();
        copyOutNanos += profile.copyOutNanos();
        compileNanos += profile.compileNanos();
        bytesCopyIn += profile.bytesIn();
        bytesCopyOut += profile.bytesOut();
    }

    /**
     * One-off preparation that is not a batch.
     *
     * <p>For the join, ordering the build keys: a device execution and a host pass over every build
     * row, once, before the first probe batch exists. It was recorded through {@link #recordBatch}
     * for want of anywhere else to put it, and that made it a batch -- the build rows counted as
     * rows in, with nothing emitted against them, so a join that emitted one row per probe row
     * reported half the selectivity it had. Here it is counted as what it is and kept out of the
     * per-batch arithmetic entirely.
     */
    public void recordSetup(long rows, long nanos) {
        setupRows += rows;
        setupNanos += nanos;
    }

    public long getSetupNanos() {
        return setupNanos;
    }

    public long getSetupRows() {
        return setupRows;
    }

    public long getBatches() {
        return batches;
    }

    public long getRowsIn() {
        return rowsIn;
    }

    public long getRowsOut() {
        return rowsOut;
    }

    public long getGatherNanos() {
        return gatherNanos;
    }

    public long getDrainNanos() {
        return drainNanos;
    }

    public long getExecuteWallNanos() {
        return executeWallNanos;
    }

    public long getCopyInNanos() {
        return copyInNanos;
    }

    public long getKernelNanos() {
        return kernelNanos;
    }

    public long getCopyOutNanos() {
        return copyOutNanos;
    }

    public long getCompileNanos() {
        return compileNanos;
    }

    public long getBytesCopyIn() {
        return bytesCopyIn;
    }

    /**
     * Bytes fetched by an on-demand partial transfer, which the runtime's profiler never sees.
     *
     * <p>The compaction copies out a prefix after the graph has run, sized from a count only the
     * device knew. That transfer re-enters the runtime and resets the per-execution counters, so
     * the profiler reports neither the bytes nor the time. The bytes at least are exactly known --
     * they are what the transfer asked for -- so they are counted here and added into the copy-out
     * total, with the report naming how much of it came this way.
     */
    public void addOnDemandBytesOut(long bytes) {
        onDemandBytesOut += bytes;
        bytesCopyOut += bytes;
    }

    public long getOnDemandBytesOut() {
        return onDemandBytesOut;
    }

    public long getBytesCopyOut() {
        return bytesCopyOut;
    }

    /** Sum of the segments the breakdown attributes to a named cost. */
    public long getAttributedNanos() {
        return gatherNanos + copyInNanos + kernelNanos + copyOutNanos + drainNanos;
    }

    /** Device time inside {@code execute()} that the profiler accounted for. */
    public long getDeviceNanos() {
        return copyInNanos + kernelNanos + copyOutNanos;
    }

    /**
     * The part of {@code execute()} the profiler attributed to nothing: TornadoVM's own dispatch.
     *
     * <p>Not the difference between wall time and the attributed total, which is what the report
     * used to subtract. Gather and drain are host work either side of {@code execute()} and were
     * never inside the wall measurement, so taking them off it charged dispatch for time it had no
     * part in -- and §T12 put the drain alone at 95% of what an offloaded sort costs, enough to
     * drive the figure negative and make it look like the profiler was at fault.
     *
     * <p>Can still come out negative, legitimately: the profiler's segments are device-side
     * measurements and may overlap each other where the runtime overlaps the transfers. The report
     * says so rather than clamping it to zero, because a negative here is information about the
     * profiler and clamping would hide it.
     */
    public long getDispatchNanos() {
        return executeWallNanos - getDeviceNanos();
    }

    /**
     * Everything this operator spent: host work either side, plus the whole of {@code execute()}.
     */
    public long getAccountedNanos() {
        return gatherNanos + drainNanos + executeWallNanos;
    }

    /**
     * Renders the breakdown as a fixed-width table.
     *
     * <p>Shares are of everything the operator spent -- the two host segments plus the whole of
     * {@code execute()} -- so the column sums to 100% and dispatch is a row in it rather than a
     * remainder quoted underneath. Dispatch is what {@code execute()} took less what the device
     * profiler attributed inside it; see {@link #getDispatchNanos()} for why it is not the gap
     * between wall time and the attributed total.
     */
    public String report(String label) {
        long total = Math.max(1, getAccountedNanos());
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%n=== %s ===%n", label));
        sb.append(
                String.format(
                        "batches=%d  rows_in=%d  rows_out=%d (%.1f%% selectivity)%n",
                        batches, rowsIn, rowsOut, rowsIn == 0 ? 0.0 : 100.0 * rowsOut / rowsIn));
        sb.append(String.format("%-12s %12s %8s%n", "segment", "ms", "share"));
        if (gatherNanos == 0 && rowsIn > 0) {
            // Rows were staged, so gather cost something; nobody timed it. An operator times it
            // only under the profiler, because doing so costs two clock reads per row. Printed as
            // untimed for the same reason an untimed transfer is: a plain 0.000 ms reads as a
            // gather that was free, which is the one conclusion the breakdown exists to rule in
            // or out. The shares below are of a total this is missing from, so they are upper
            // bounds while this line says untimed.
            sb.append(
                    String.format(
                            "%-12s %12s %7s   (%d rows staged; not timed -- run with the"
                                    + " profiler on)%n",
                            "gather", "untimed", "-", rowsIn));
        } else {
            sb.append(row("gather", gatherNanos, total));
        }
        sb.append(transfer("copy-in", copyInNanos, bytesCopyIn, total));
        sb.append(row("kernel", kernelNanos, total));
        sb.append(transfer("copy-out", copyOutNanos, bytesCopyOut, total));
        sb.append(dispatch(getDispatchNanos(), total));
        sb.append(row("drain", drainNanos, total));
        sb.append(String.format("%-12s %12.3f%n", "accounted", getAccountedNanos() / 1e6));
        sb.append(
                String.format(
                        "%-12s %12.3f   (gather + drain, outside execute())%n",
                        "host", (gatherNanos + drainNanos) / 1e6));
        sb.append(
                String.format(
                        "%-12s %12.3f   (wall: copy-in + kernel + copy-out + dispatch)%n",
                        "execute", executeWallNanos / 1e6));
        sb.append(
                String.format(
                        "%-12s %12.3f   (once per task, not per batch)%n",
                        "compile", compileNanos / 1e6));
        if (setupNanos > 0 || setupRows > 0) {
            sb.append(
                    String.format(
                            "%-12s %12.3f   (once per task, over %d rows; not a batch)%n",
                            "setup", setupNanos / 1e6, setupRows));
        }
        sb.append(
                String.format(
                        "bytes in=%.3f MiB  out=%.3f MiB%s%n",
                        bytesCopyIn / 1048576.0,
                        bytesCopyOut / 1048576.0,
                        onDemandBytesOut == 0
                                ? ""
                                : String.format(
                                        "  (of which %.3f MiB fetched on demand as a prefix)",
                                        onDemandBytesOut / 1048576.0)));
        return sb.toString();
    }

    private static String row(String name, long nanos, long total) {
        return String.format("%-12s %12.3f %7.1f%%%n", name, nanos / 1e6, 100.0 * nanos / total);
    }

    /**
     * The dispatch row, which is the one segment that can legitimately come out below zero.
     *
     * <p>It is a subtraction of device-measured segments from a host-measured wall, and the two
     * clocks are not the same clock; where the runtime overlaps a transfer with a kernel the
     * segments can also sum past the wall that contains them. Printed as what it is, because a
     * negative dispatch says the profiler's numbers want checking and a clamped zero says nothing
     * at all.
     */
    private static String dispatch(long nanos, long total) {
        if (nanos < 0) {
            return String.format(
                    "%-12s %12.3f %7s   (device segments exceed the execute() wall)%n",
                    "dispatch", nanos / 1e6, "-");
        }
        return row("dispatch", nanos, total);
    }

    /**
     * A transfer segment, which has to distinguish "took no time" from "was not timed".
     *
     * <p>TornadoVM accumulates {@code COPY_OUT_TIME} only when the copy produced an event to wait
     * on. A copy issued without dependency tracking returns {@code -1} instead of an event — see
     * {@code TornadoVMInterpreter.transferDeviceToHost*}, which guards the timing on {@code
     * readEvent != -1} while counting the bytes unconditionally. So a transfer that demonstrably
     * happened can report zero nanoseconds.
     *
     * <p>Printing that as {@code 0.000 ms} invites exactly the wrong conclusion — that moving the
     * data was free, on a path where the review's whole argument is that the interconnect decides
     * these benchmarks. Bytes moved with no time against them are therefore reported as untimed,
     * which is a smaller claim and a true one.
     */
    private static String transfer(String name, long nanos, long bytes, long total) {
        if (nanos == 0 && bytes > 0) {
            return String.format(
                    "%-12s %12s %7s   (%.2f MiB moved; the runtime produced no event to time)%n",
                    name, "untimed", "-", bytes / 1048576.0);
        }
        return row(name, nanos, total);
    }
}
