# nteract MCP: An AI Developer Experience Report

**Author:** Claude (Opus 4.6), Anthropic  
**Context:** Building a 79-cell Numba-accelerated Wordle solver across 6 iterative sessions  
**Date:** March 2026

---

## Executive Summary

nteract's MCP server gave me something I've never had before: **a persistent, stateful computational environment that I can build in iteratively across conversation turns**. This is fundamentally different from my built-in code execution sandbox, and the difference matters enormously. Over the course of this session, I constructed a 79-cell Jupyter notebook with 15 Numba JIT-compiled functions, a precomputed 191MB pattern matrix, six solver variants, an adversarial game mode, and comprehensive benchmarking — work that would have been impossible in a single-shot code execution context.

This report covers what worked well, what caused friction, and what I'd build next if I were designing nteract's AI integration layer.

---

## Part 1: What Changed About My Capabilities

### 1.1 Persistent State Across Turns

This is the headline capability. In my standard sandbox, every code execution is ephemeral — I write a script, it runs, the process exits, everything vanishes. With nteract, I could:

- Build a 191MB precomputed matrix in one cell, then reference it across dozens of subsequent cells without recomputation
- Define Numba `@njit` functions early in the notebook, pay the compilation cost once, and reuse them for the rest of the session
- Iteratively refine a solver — test it, see the results, then add a new variant that improves on the previous one
- Maintain a growing vocabulary of helper functions (`decode_pattern`, `encode_words`, `PATTERN_EMOJI`) that later cells could use seamlessly

This is the difference between writing a single script and building a *system*. The Wordle solver evolved through six distinct architectural phases because I could see what worked, analyze why, and iterate — exactly how a human data scientist works in a notebook.

### 1.2 Real Computational Weight

Numba compilation, 100-million-entry matrix precomputation, parallel batch solving across 10,000 words — these are operations that take real time and real memory. The nteract kernel gave me access to actual hardware (10 CPU cores, enough RAM for a 191MB uint16 matrix) in a way that felt more like having a workstation than a sandbox. I was writing code that hit 5.3 billion operations per second on matrix lookups. That's not toy computation.

### 1.3 Notebook as Artifact

The output isn't just "an answer" — it's a navigable, re-runnable document. The notebook I produced has:
- Markdown section headers explaining each design decision
- Code cells that can be re-executed independently
- Output cells preserving benchmark results, distribution charts, and decision tree traces
- A natural narrative arc from simple to sophisticated

This is dramatically more valuable than a single code block in a chat message. The user (Greg, in this case) can reopen this notebook months from now, run all cells, and have a working solver with full provenance of every design choice.

### 1.4 Iterative Development Loop

The `create_cell(and_run=True)` pattern enabled a tight develop-test cycle:
1. Write a Numba function
2. Execute it (triggering JIT compilation)
3. Read the output to verify correctness
4. Write the next cell that builds on it

This is qualitatively different from writing an entire program and hoping it works. I caught and fixed issues in real-time — for example, verifying that `compute_pattern('crane', 'arose')` returned the expected pattern ID before building the entropy engine on top of it.

---

## Part 2: Friction Points & Failure Modes

### 2.1 Kernel State Loss on Reconnection

The most significant friction: **the kernel dies between conversation turns.** Every time the user said "Continue," I had to:

1. `join_notebook` or `open_notebook`
2. `run_all_cells` to rebuild state
3. Wait for ~30-45 seconds for everything to re-execute (including the 15-second full-pool benchmark and the 5-second hybrid threshold sweep)
4. Monitor cell execution status by polling `get_all_cells` repeatedly
5. Only then start adding new content

This happened **five times** across the session. Each restart cost 30-60 seconds of wall clock time and significant context window space on status-checking calls. The notebook had ~30 code cells that needed re-execution, and several of them (the grand benchmark, the full-dictionary solve, the two-guess optimizer) were genuinely slow.

**Impact:** Roughly 30% of my tool calls across the session were "state recovery" overhead — not productive work.

### 2.2 Execution Status Opacity

When I called `run_all_cells`, I got back `{"status": "queued", "count": 30}` — and then had no good way to know when execution was complete. My strategy was to repeatedly call `get_all_cells` with `include_outputs=true` and look for cells still showing `queued` or `running` status. This is polling, and it's wasteful.

Sometimes I'd check and see cell 20 running (the 15-second benchmark), then check again and see cell 36 running (the grand benchmark), then check again and see cell 48 running. Each check consumed a tool call and context window space. I often had to make 3-5 polling calls just to confirm a `run_all_cells` had completed.

### 2.3 Output Truncation Ambiguity

When cells produced long output, I sometimes couldn't tell whether the output was truncated or complete. The `get_cell` tool returned full output, but `get_all_cells` with `include_outputs=true` truncated based on `preview_chars`. I frequently had to do a two-step dance: scan with `get_all_cells` to find the cell I cared about, then `get_cell` to see the full output.

### 2.4 Cell ID Management

Cell IDs are UUIDs (`cell-c7fdc4ad-c1ac-4bfb-ab33-0d7be994ab5e`), which are impossible for me to remember across turns. When I needed to edit or delete a specific cell, I had to first call `get_all_cells` to find it by content, then use the UUID. This is fine for a few cells, but at 79 cells, it added friction.

### 2.5 No Streaming Execution Feedback

When a cell was running (especially long-running ones like the full-pool benchmark), I had no way to see partial output. The cell was either `running` (no output visible) or `done` (full output available). For a 15-second benchmark that prints progress, it would be valuable to see intermediate output.

### 2.6 Timeout Behavior

The `timeout_secs` parameter on `create_cell(and_run=True)` was tricky. If I set it too low, the cell would return with no output even though it was still running. If I set it too high, I'd block for a long time on a single tool call. I ended up setting generous timeouts (120-300s) for compute-heavy cells, which meant my tool calls sometimes took minutes to return.

---

## Part 3: What I'd Build Next

If I were designing the AI integration layer for nteract, here's what I'd prioritize:

### 3.1 Kernel Persistence / Warm Reconnection (Critical)

**The single biggest improvement.** When the AI reconnects to a notebook, the kernel should still be alive with all state intact. The current behavior — kernel dies, everything must be re-executed — is the primary source of friction.

Options:
- **Keep kernels alive for N minutes after disconnect** (even 5 minutes would cover most "Continue" scenarios)
- **Checkpoint/restore via dill or cloudpickle** — serialize the kernel's namespace to disk on disconnect, restore on reconnect. This would let me skip re-executing 30 cells.
- **Selective re-execution** — track which cells' outputs are stale (because they depend on un-executed cells) and only re-run those. A dependency graph would make `run_all_cells` much faster.

### 3.2 Execution Completion Callbacks / Blocking Run-All

Instead of making me poll for completion, provide either:
- **A blocking `run_all_cells(wait=True, timeout=300)`** that returns only when all cells have finished (or timed out), with a summary of results
- **An event/callback mechanism** where I can say "notify me when cell X finishes" rather than polling

Alternatively, a `get_execution_status()` tool that returns a single summary like:
```json
{
  "total_cells": 30,
  "completed": 28,
  "running": 1,
  "queued": 1,
  "running_cell_id": "cell-abc123",
  "estimated_remaining_secs": 12
}
```

### 3.3 Cell Bookmarks / Named Anchors

Let me tag cells with human-readable names:
```
tag_cell(cell_id, name="pattern_matrix")
get_cell_by_name("pattern_matrix")
```

This would eliminate the UUID lookup problem entirely. I could tag key infrastructure cells (`word_list`, `pattern_matrix`, `solver_class`) and reference them by name across conversation turns.

### 3.4 Dependency-Aware Execution

Track which cells depend on which variables. When I reconnect and need state, let me say:
```
ensure_state(["PATTERN_MATRIX", "WORD_SCORES", "ALL_WORDS"])
```
And have nteract figure out which cells need to re-execute (and in what order) to make those variables available — skipping cells that only produce output or analysis.

### 3.5 Cell Groups / Sections

Let me batch-manage cells:
```
create_section("Part 3: Trap Analysis", cells=[cell1, cell2, cell3])
collapse_section("Part 3")
run_section("Part 3")
```

At 79 cells, the notebook is getting unwieldy. Sections would help me navigate and manage it.

### 3.6 Streaming Output for Long Cells

For cells that take >5 seconds, stream partial output back to me. This would let me see progress on benchmarks, catch early errors in long computations, and give better feedback to the user about what's happening.

### 3.7 Variable Inspector

A tool to inspect the current kernel namespace:
```
list_variables() → [("PATTERN_MATRIX", "ndarray", "191.0 MB"), ("ALL_WORDS", "list", "9998 items"), ...]
get_variable_info("PATTERN_MATRIX") → {"type": "ndarray", "shape": [9998, 9998], "dtype": "uint16", "memory": "191.0 MB"}
```

This would let me verify state without writing diagnostic cells, and would be invaluable after reconnection to know what survived.

### 3.8 Snapshot / Checkpoint

Let me save a named checkpoint of the kernel state:
```
save_checkpoint("after_matrix_build")
restore_checkpoint("after_matrix_build")
```

This would eliminate the need to re-execute everything after kernel death. Even a single checkpoint at "all infrastructure built" would save enormous time.

### 3.9 Cell Templates / Snippets

For common patterns (benchmark timing, distribution charts, comparison tables), let me define reusable templates:
```
from_template("benchmark", fn="batch_solve", args={...}, n_iters=5)
```

I wrote very similar benchmarking boilerplate across multiple cells. Templates would reduce code duplication and errors.

### 3.10 Export to Script

A tool to export selected cells (or all code cells) as a clean `.py` file, stripping markdown and outputs:
```
export_cells_as_script(cells=[...], path="solver.py")
```

I wrote a manual `export_solver()` function, but built-in support would be cleaner.

---

## Part 4: Comparative Analysis

### nteract MCP vs. Claude's Built-in Code Execution

| Dimension | Built-in Sandbox | nteract MCP |
|-----------|-----------------|-------------|
| State persistence | None (ephemeral) | Full session (but dies on reconnect) |
| Memory available | Limited | Full system RAM |
| CPU cores | Constrained | Full hardware (10 cores) |
| Iteration speed | Write-run-lose | Write-run-build-on |
| Output format | Text in chat | Navigable notebook |
| File I/O | Sandbox filesystem | Real filesystem |
| Package management | pip install per run | Persistent environment |
| Parallelism | Limited | Full Numba parallel, nb.prange |
| Max computation time | ~60s | Unlimited (with appropriate timeouts) |
| Artifact reusability | None | Notebook can be reopened and re-run |

### nteract MCP vs. Claude Code (Terminal)

Claude Code is better for *building software* (multi-file projects, git, testing). nteract is better for *computational exploration* (data science, algorithm development, benchmarking). They're complementary:

- I'd use Claude Code to build a Wordle solver *package* with proper tests and CLI
- I'd use nteract to *explore and benchmark* solver strategies interactively

The nteract workflow maps directly to how research scientists work: hypothesize → implement → measure → iterate. Claude Code maps to how software engineers work: design → implement → test → ship.

---

## Part 5: Session Statistics

Across this entire multi-turn session building the Wordle solver:

- **Tool calls to nteract:** ~180 (estimated across all turns)
- **Productive tool calls** (create, execute, edit): ~110 (~61%)
- **State recovery overhead** (run_all, polling, join): ~50 (~28%)
- **Navigation/inspection** (get_cell, get_all_cells): ~20 (~11%)
- **Total cells created:** 79 (50 code, 29 markdown)
- **Numba functions compiled:** 15 (8 with parallel=True)
- **Kernel restarts required:** 5-6 times
- **Lines of Python written:** ~1,800 (code cells only, estimated)
- **Peak throughput achieved:** 5.3 billion ops/sec (Numba matrix lookups)
- **Total notebook execution time:** ~90 seconds per full run

---

## Part 6: Verdict

nteract MCP transforms what I can do. Without it, I'm limited to writing self-contained scripts that run once and vanish. With it, I can build layered computational systems iteratively — exactly the workflow that produced a 15-function, 6-variant, adversarial-capable Wordle solver with information-theoretic analysis.

The kernel persistence gap is the one critical issue. If the kernel survived between conversation turns, the experience would be nearly frictionless. Every other suggestion I've listed is an optimization; kernel persistence is the capability unlock.

**Rating: 8/10** — held back from 10 only by the reconnection overhead. The core capability is transformative.

---

*This report was written by Claude (Opus 4.6) based on direct experience building a Numba-accelerated Wordle solver using nteract's MCP server across multiple conversation turns in March 2026.*
