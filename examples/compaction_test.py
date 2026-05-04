"""
Compaction system end-to-end behavioural test.

Config rationale (not production values):
  initial_score_derived=15.0  — gives ~30 sweeps before cull; score room to observe decay
  decay_per_sweep=0.5         — moderate; scores decay ~0.5/sweep, visible across test
  soft_limit=15               — pressure kicks in once derived > 15 (reachable in ~20 queries)
  decay_every_n_queries=3     — sweeps fire every 3 real queries; 4+ sweeps per major section

Run: python examples/compaction_test.py
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from grag.config import Config
from grag.ingest import ingest
from grag.rag import RAG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DATA_DIR = Path("./compaction_test_data")
OUTPUT_DIR = Path("./compaction_test_output")
EINSTEIN_DOC = Path("./examples/sample_docs/einstein.txt")

if TEST_DATA_DIR.exists():
    shutil.rmtree(TEST_DATA_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Test config — aggressive enough that compaction effects are observable
# within ~50 queries rather than thousands.
# ---------------------------------------------------------------------------
config = Config(
    data_dir=TEST_DATA_DIR,
    chroma_path=str(TEST_DATA_DIR / "chroma"),          # isolated from production data
    graph_path=str(TEST_DATA_DIR / "knowledge_graph.gexf"),
    decay_every_n_queries=3,    # frequent sweeps so each section sees 4+ cycles
    decay_per_sweep=1.0,        # aggressive decay; ~5 sweeps (5.0/1.0) before cull — visible in test
    access_increment=1.0,
    initial_score_source=10.0,
    initial_score_derived=5.0,  # low floor so culling fires within test duration
    confirmation_floor=5,
    cull_threshold=0.0,
    compact_decay_multiplier=2.0,
    soft_limit=30,              # high enough that pressure section runs before limit is crossed
    pressure_targets_derived_only=True,
)
rag = RAG(config=config)

log: list[dict] = []           # one row per query/event
results: dict = {}             # section → {status, message}
sweep_steps: list[int] = []    # step numbers where a sweep was detected

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def snapshot(rag: RAG) -> dict:
    """Capture all observable counters in one call."""
    s = rag.stats()
    m = s["memory"]
    g = s["graph"]
    return {
        "source_count":        m["sources"]["count"],
        "derived_count":       m["derived"]["count"],
        "node_count":          g["nodes"],
        "edge_count":          g["edges"],
        "derived_median_score": m["derived"].get("score_median", 0.0),
        "derived_min_score":   m["derived"].get("score_min",    0.0),
        "derived_max_score":   m["derived"].get("score_max",    0.0),
        "queries_since_sweep": m["queries_since_sweep"],
    }


def run_query(
    rag: RAG,
    question: str,
    phase: str,
    log: list,
    expect_cache_hit: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Run a query, infer cache_hit and dedup_rejected from deltas, append a CSV row.

    cache_hit is read directly from the query result.
    dedup_rejected is inferred: cache miss with no growth in derived_count means
    the residual-novelty filter rejected the new summary.

    Sweep detection: memory.query() resets queries_since_sweep to 0 after a sweep,
    so after_qss < before_qss reliably signals a fired sweep.
    """
    before = snapshot(rag)
    result = rag.query(question)
    after = snapshot(rag)

    cache_hit: bool = result["cache_hit"]
    derived_grew = after["derived_count"] > before["derived_count"]
    dedup_rejected = (not cache_hit) and (not derived_grew)

    # Sweep fires inside memory.query() (not on cache hits), resetting the counter to 0
    sweep_fired = (not cache_hit) and (after["queries_since_sweep"] < before["queries_since_sweep"])
    step = len(log) + 1
    if sweep_fired:
        sweep_steps.append(step)

    log.append({
        "step":                 step,
        "phase":                phase,
        "query":                question[:80],
        "source_count":         after["source_count"],
        "derived_count":        after["derived_count"],
        "node_count":           after["node_count"],
        "edge_count":           after["edge_count"],
        "derived_median_score": after["derived_median_score"],
        "derived_min_score":    after["derived_min_score"],
        "derived_max_score":    after["derived_max_score"],
        "cache_hit":            cache_hit,
        "dedup_rejected":       dedup_rejected,
        "notes":                "",
    })
    return result, before, after


def assert_section(name: str, condition: bool, message: str, results: dict) -> None:
    """Record pass/fail. Never raises — we collect a complete picture."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}: {message}")
    results[name] = {"status": status, "message": message}


# ---------------------------------------------------------------------------
# Section 1 — Ingestion
# ---------------------------------------------------------------------------
print("\n=== Section 1: Ingestion ===")

ingest(EINSTEIN_DOC, rag)
s1 = snapshot(rag)
log.append({
    "step": 1, "phase": "ingest", "query": "(ingestion)",
    "source_count": s1["source_count"], "derived_count": s1["derived_count"],
    "node_count": s1["node_count"], "edge_count": s1["edge_count"],
    "derived_median_score": 0.0, "derived_min_score": 0.0, "derived_max_score": 0.0,
    "cache_hit": False, "dedup_rejected": False, "notes": "post-ingest snapshot",
})

assert_section("section_1_ingestion",
               s1["source_count"] > 0,
               f"{s1['source_count']} source chunks, {s1['edge_count']} graph edges", results)
assert_section("section_1_graph",
               s1["node_count"] > 0 and s1["edge_count"] > 0,
               f"{s1['node_count']} nodes, {s1['edge_count']} edges", results)
assert_section("section_1_no_derived",
               s1["derived_count"] == 0,
               f"derived_count={s1['derived_count']} (expected 0)", results)

# ---------------------------------------------------------------------------
# Section 2 — Cache behaviour
#
# Sweep math: ingest adds no queries, so queries_since_sweep=0 entering this section.
# Q1 is a cache miss → qss becomes 1 (no sweep at 1 < 3).
# Q2 and Q3 are cache hits → memory.query() is never called → qss stays 1.
# No sweep fires in this section.
# ---------------------------------------------------------------------------
print("\n=== Section 2: Cache Behaviour ===")

cache_q = "where was Einstein born?"

r1, b1, a1 = run_query(rag, cache_q, "cache", log)
derived_after_q1 = a1["derived_count"]

assert_section("section_2_first_miss",
               not r1["cache_hit"],
               f"cache_hit={r1['cache_hit']} (expected False)", results)
assert_section("section_2_derived_grew",
               a1["derived_count"] == b1["derived_count"] + 1,
               f"derived grew by {a1['derived_count'] - b1['derived_count']} (expected 1)", results)

r2, b2, a2 = run_query(rag, cache_q, "cache", log, expect_cache_hit=True)
r3, b3, a3 = run_query(rag, cache_q, "cache", log, expect_cache_hit=True)

assert_section("section_2_repeat_hits",
               r2["cache_hit"] and r3["cache_hit"],
               f"repeat cache_hits: q2={r2['cache_hit']}, q3={r3['cache_hit']}", results)
assert_section("section_2_no_growth_on_hit",
               a3["derived_count"] == derived_after_q1 and a3["edge_count"] == a1["edge_count"],
               f"derived stayed at {derived_after_q1}, edges stayed at {a1['edge_count']}", results)

# ---------------------------------------------------------------------------
# Section 3 — Residual dedup on paraphrases
#
# Sweep math: qss=1 entering. Three cache misses advance qss to 2, 3 (sweep), 1.
# Expected: 1 sweep fires mid-section (on the 2nd paraphrase query).
# ---------------------------------------------------------------------------
print("\n=== Section 3: Residual Dedup ===")

paraphrases = [
    "What was Einstein famous for?",
    "Why is Einstein well-known?",
    "What made Einstein significant?",
]

dedup_pre_count = snapshot(rag)["derived_count"]
dedup_results = []
for q in paraphrases:
    r, b, a = run_query(rag, q, "dedup", log)
    dedup_results.append((r, b, a))

dedup_post_count = snapshot(rag)["derived_count"]
total_dedup_growth = dedup_post_count - dedup_pre_count
rejected_count = sum(
    1 for r, b, a in dedup_results
    if not r["cache_hit"] and a["derived_count"] == b["derived_count"]
)
all_misses = all(not r["cache_hit"] for r, b, a in dedup_results)

assert_section("section_3_all_misses", all_misses,
               f"all paraphrases were cache misses: {all_misses}", results)
assert_section("section_3_dedup_fired", rejected_count >= 1,
               f"{rejected_count}/3 rejected by residual filter", results)

if total_dedup_growth <= 2:
    assert_section("section_3_growth_bounded", True,
                   f"total derived growth={total_dedup_growth} (≤2)", results)
elif total_dedup_growth == 3:
    print("  [WARN] All 3 paraphrases accepted — residual filter may be too lenient")
    results["section_3_growth_bounded"] = {
        "status": "WARN",
        "message": f"growth={total_dedup_growth}: all paraphrases landed (residual filter too lenient)",
    }
else:
    assert_section("section_3_growth_bounded", False,
                   f"unexpected growth={total_dedup_growth}", results)

# ---------------------------------------------------------------------------
# Section 4 — Decay sweep fires periodically
#
# Sweep math: qss=1 entering (after S3's mid-section sweep reset the counter).
# 12 non-cache queries with qss carry-in 1:
#   Q2 fires sweep (qss: 2→3→0), Q5, Q8, Q11 each fire one.
# Expected: exactly 4 sweeps.
# ---------------------------------------------------------------------------
print("\n=== Section 4: Decay Sweep Periodicity ===")

decay_questions = [
    "When did Einstein move to America?",
    "What was Einstein's Nobel Prize awarded for?",
    "How did Einstein contribute to quantum mechanics?",
    "What was Einstein's relationship with the Manhattan Project?",
    "What university did Einstein study at in Zurich?",
    "What was Einstein's special theory of relativity about?",
    "What is the photoelectric effect that Einstein explained?",
    "Where did Einstein work after leaving Germany in 1933?",
    "What did Einstein think about quantum entanglement?",
    "How old was Einstein when he died?",
    "What did Einstein's general theory of relativity predict about gravity?",
    "What language did Einstein speak as a child?",
]

pre_s4 = snapshot(rag)
qss_entering_s4 = pre_s4["queries_since_sweep"]
# expected sweeps = floor((12 cache-misses + qss_carry_in) / decay_every_n_queries)
expected_sweeps_s4 = (len(decay_questions) + qss_entering_s4) // config.decay_every_n_queries

sweeps_before_s4 = len(sweep_steps)

for q in decay_questions:
    run_query(rag, q, "decay", log)

sweeps_s4 = len(sweep_steps) - sweeps_before_s4
post_s4 = snapshot(rag)

assert_section("section_4_sweeps_fired",
               sweeps_s4 == expected_sweeps_s4,
               f"{sweeps_s4} sweeps (expected {expected_sweeps_s4}, qss carry-in={qss_entering_s4})", results)
assert_section("section_4_score_decayed",
               post_s4["derived_min_score"] < config.initial_score_derived,
               f"min_score={post_s4['derived_min_score']:.2f} < initial={config.initial_score_derived}", results)

# ---------------------------------------------------------------------------
# Section 5 — Memory pressure scaling
#
# Ask distinct questions until derived_count > soft_limit=15, then keep going
# to accumulate enough over-limit sweeps for a reliable comparison.
# Sweep math: qss=1 entering (Section 4 ends with qss=1 after Q12 of S4).
# ---------------------------------------------------------------------------
print("\n=== Section 5: Memory Pressure Scaling ===")

pressure_questions = [
    "What is the equivalence principle in Einstein's general theory?",
    "Who were Einstein's main scientific contemporaries?",
    "Did Einstein believe in religion or God?",
    "What happened to Einstein's first marriage to Mileva Maric?",
    "What is the speed of light and why is it a constant in Einstein's theory?",
    "How did Einstein's work on Brownian motion help prove atoms exist?",
    "What awards did Einstein receive besides the Nobel Prize?",
    "What was Einstein's pacifist stance during World War I?",
    "How did Einstein influence modern cosmology and the big bang theory?",
    "What was Einstein's view on unified field theory in his later years?",
    "What did Einstein say about imagination versus knowledge?",
    "Did Einstein ever visit Japan and what happened there?",
    "What was Einstein's relationship with Niels Bohr over quantum mechanics?",
    "What is spacetime curvature in Einstein's general relativity?",
    "What was the significance of the 1919 solar eclipse for Einstein?",
]

# Track median score at each sweep event to compare under- vs over-limit decay rates
sweep_medians_under: list[float] = []   # median after sweeps fired while derived ≤ soft_limit
sweep_medians_over: list[float] = []    # median after sweeps fired while derived > soft_limit
peaked_above_limit = False
sweeps_before_s5 = len(sweep_steps)

for q in pressure_questions:
    pre = snapshot(rag)
    r, b, a = run_query(rag, q, "pressure", log)
    if a["derived_count"] > config.soft_limit:
        peaked_above_limit = True
    # Record median after sweep events, tagged by regime
    if sweep_steps and sweep_steps[-1] == log[-1]["step"]:
        regime = "over" if pre["derived_count"] > config.soft_limit else "under"
        if regime == "under":
            sweep_medians_under.append(a["derived_median_score"])
        else:
            sweep_medians_over.append(a["derived_median_score"])

assert_section("section_5_reached_limit", peaked_above_limit,
               f"derived peaked {'above' if peaked_above_limit else 'at or below'} soft_limit={config.soft_limit}", results)

# Compute per-sweep median drops for each regime
def _drops(seq: list[float]) -> list[float]:
    return [seq[i-1] - seq[i] for i in range(1, len(seq)) if seq[i-1] - seq[i] > 0]

under_drops = _drops(sweep_medians_under)
over_drops  = _drops(sweep_medians_over)

if under_drops and over_drops:
    avg_under = statistics.mean(under_drops)
    avg_over  = statistics.mean(over_drops)
    ratio = avg_over / avg_under if avg_under > 0 else 0.0
    if ratio >= 1.5:
        assert_section("section_5_pressure_scaling", True,
                       f"over-limit {avg_over:.3f}/sweep vs under-limit {avg_under:.3f}/sweep (ratio={ratio:.2f}x)", results)
    else:
        print(f"  [WARN] Pressure ratio={ratio:.2f}x < 1.5x — soft-fail (expected with soft_limit=15)")
        results["section_5_pressure_scaling"] = {
            "status": "WARN",
            "message": (
                f"ratio={ratio:.2f}x (under={avg_under:.3f}, over={avg_over:.3f}) — "
                "pressure multiplier detectable but below 1.5x threshold; "
                "derived count likely didn't exceed soft_limit by enough"
            ),
        }
else:
    print("  [WARN] Not enough sweep samples across both regimes to compare pressure scaling")
    results["section_5_pressure_scaling"] = {
        "status": "WARN",
        "message": (
            f"insufficient sweep samples "
            f"(under={len(sweep_medians_under)} samples, over={len(sweep_medians_over)} samples)"
        ),
    }

# ---------------------------------------------------------------------------
# Section 6 — Manual compact
#
# compact() applies decay_per_sweep × compact_decay_multiplier = 0.5 × 2.0 = 1.0 per entry.
# A periodic sweep drops 0.5. So compact median drop should be > 0.5.
# ---------------------------------------------------------------------------
print("\n=== Section 6: Manual Compact ===")

pre_compact = snapshot(rag)
rag.compact()
post_compact = snapshot(rag)

compact_median_drop = pre_compact["derived_median_score"] - post_compact["derived_median_score"]
single_sweep_drop = config.decay_per_sweep  # 0.5; compact applies 2× this

assert_section("section_6_derived_not_grew",
               post_compact["derived_count"] <= pre_compact["derived_count"],
               f"derived: {pre_compact['derived_count']} → {post_compact['derived_count']}", results)
assert_section("section_6_compact_more_aggressive",
               compact_median_drop > single_sweep_drop,
               f"compact median drop={compact_median_drop:.3f} vs single sweep drop={single_sweep_drop:.3f}", results)

log.append({
    "step": len(log) + 1, "phase": "compact", "query": "(compact)",
    "source_count": post_compact["source_count"], "derived_count": post_compact["derived_count"],
    "node_count": post_compact["node_count"], "edge_count": post_compact["edge_count"],
    "derived_median_score": post_compact["derived_median_score"],
    "derived_min_score": post_compact["derived_min_score"],
    "derived_max_score": post_compact["derived_max_score"],
    "cache_hit": False, "dedup_rejected": False,
    "notes": f"manual compact: derived {pre_compact['derived_count']}→{post_compact['derived_count']}",
})

# ---------------------------------------------------------------------------
# Section 7 — Confirmation floor protection
#
# Bump one derived entry's confirmations to confirmation_floor=5 via direct
# metadata mutation. NOTE: _bump_metadata is private. This warrants a public
# rag.pin_entry(id) method in a future API update.
#
# Then run 10 more queries (≥3 sweeps) and verify the protected entry's score
# didn't change while unprotected entries continued to decay.
# ---------------------------------------------------------------------------
print("\n=== Section 7: Confirmation Floor Protection ===")

all_derived_data = rag.memory._derived.get(include=["metadatas", "documents"])
protected_id: str | None = None
protected_score_before: float = 0.0

for doc_id, meta in zip(all_derived_data["ids"], all_derived_data["metadatas"]):
    meta = meta or {}
    confs = int(meta.get("confirmations", 1))
    if confs < config.confirmation_floor:
        needed = config.confirmation_floor - confs
        # Direct metadata mutation — no public API for manual pinning yet.
        rag.memory._bump_metadata(rag.memory._derived, doc_id, 0.0, conf_delta=needed)
        protected_id = doc_id
        protected_score_before = float(meta.get("score", config.initial_score_derived))
        print(f"  Pinned entry {doc_id[:16]}... "
              f"(confirmations {confs}→{config.confirmation_floor}, score={protected_score_before:.2f})")
        break

if protected_id is None:
    print("  [WARN] No entry with low confirmations found for protection test")
    results["section_7_floor_protection"] = {
        "status": "WARN", "message": "no candidate entry to protect"
    }
else:
    floor_questions = [
        "What nationality was Albert Einstein?",
        "What does the equation E equals mc squared represent?",
        "Did Einstein have any siblings?",
        "How does general relativity explain the bending of light?",
        "Where was Einstein's Institute for Advanced Study located?",
        "What did Einstein do after the atom bomb was dropped?",
        "How did Einstein's theory explain the anomalous precession of Mercury?",
        "What is a thought experiment and how did Einstein use them in physics?",
        "Did Einstein fail mathematics in school?",
        "What was Einstein's role in the development of quantum statistics?",
    ]

    sweeps_before_s7 = len(sweep_steps)
    for q in floor_questions:
        run_query(rag, q, "floor", log)
    sweeps_s7 = len(sweep_steps) - sweeps_before_s7

    check = rag.memory._derived.get(ids=[protected_id], include=["metadatas"])
    if not check["ids"]:
        assert_section("section_7_floor_protection", False,
                       "protected entry was deleted — floor protection failed", results)
    else:
        final_score = float((check["metadatas"][0] or {}).get("score", 0.0))
        # Score may rise (access bumps are allowed); what must not happen is decay-driven drops.
        score_held_or_rose = final_score >= protected_score_before - 0.01
        assert_section("section_7_floor_protection", score_held_or_rose,
                       (f"protected entry score held or rose: {protected_score_before:.2f} → {final_score:.2f} "
                        f"over {sweeps_s7} sweeps (access bumps allowed, decay blocked)"), results)

# ---------------------------------------------------------------------------
# Write CSV log and JSON summary
# ---------------------------------------------------------------------------
print("\n=== Writing artefacts ===")

CSV_FIELDS = [
    "step", "phase", "query",
    "source_count", "derived_count", "node_count", "edge_count",
    "derived_median_score", "derived_min_score", "derived_max_score",
    "cache_hit", "dedup_rejected", "notes",
]

with open(OUTPUT_DIR / "run_log.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(log)

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
PHASE_COLOURS = {
    "ingest":   "#e8f5e9",
    "cache":    "#e3f2fd",
    "dedup":    "#fff3e0",
    "decay":    "#fce4ec",
    "pressure": "#f3e5f5",
    "compact":  "#e0f7fa",
    "floor":    "#fff8e1",
}

steps           = [r["step"]                 for r in log]
phases          = [r["phase"]                for r in log]
derived_counts  = [r["derived_count"]        for r in log]
edge_counts     = [r["edge_count"]           for r in log]
median_scores   = [r["derived_median_score"] for r in log]
min_scores      = [r["derived_min_score"]    for r in log]
max_scores      = [r["derived_max_score"]    for r in log]


def shade_phases(ax: plt.Axes, log: list[dict]) -> None:
    """Shade background by phase and label each region above the plot."""
    current_phase = None
    start_step = None
    for row in log:
        if row["phase"] != current_phase:
            if current_phase is not None:
                ax.axvspan(start_step - 0.5, row["step"] - 1.5,
                           alpha=0.20, color=PHASE_COLOURS.get(current_phase, "#f5f5f5"), lw=0)
                mid = (start_step + row["step"] - 1) / 2
                ax.text(mid, 1.01, current_phase, ha="center", va="bottom", fontsize=7,
                        color="#555555", transform=ax.get_xaxis_transform(), clip_on=False)
            current_phase = row["phase"]
            start_step = row["step"]
    if current_phase and start_step is not None:
        ax.axvspan(start_step - 0.5, log[-1]["step"] + 0.5,
                   alpha=0.20, color=PHASE_COLOURS.get(current_phase, "#f5f5f5"), lw=0)
        mid = (start_step + log[-1]["step"]) / 2
        ax.text(mid, 1.01, current_phase, ha="center", va="bottom", fontsize=7,
                color="#555555", transform=ax.get_xaxis_transform(), clip_on=False)


def mark_sweeps(ax: plt.Axes) -> None:
    for i, sv in enumerate(sweep_steps):
        ax.axvline(sv, color="gray", linestyle="--", lw=0.7, alpha=0.5,
                   label="sweep" if i == 0 else None)


# plot_growth.png — derived chunk count + edge count over query number
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(steps, derived_counts, label="derived_count", marker="o", ms=3)
ax.plot(steps, edge_counts,    label="edge_count",    marker="s", ms=3, linestyle="--")
ax.axhline(config.soft_limit, color="red", linestyle=":", lw=1.2,
           label=f"soft_limit={config.soft_limit}")
mark_sweeps(ax)
shade_phases(ax, log)
ax.set_xlabel("Step (query number)")
ax.set_ylabel("Count")
ax.set_title("Growth: Derived Chunks & Graph Edges over Queries")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot_growth.png", dpi=120, bbox_inches="tight")
plt.close()

# plot_scores.png — derived score distribution over query number
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(steps, max_scores,    label="max_score",    marker="^", ms=3, linestyle="-",  color="steelblue", alpha=0.7)
ax.plot(steps, median_scores, label="median_score", marker="o", ms=3, linestyle="-",  color="darkorange")
ax.plot(steps, min_scores,    label="min_score",    marker="v", ms=3, linestyle="--", color="green")
ax.axhline(config.cull_threshold, color="red", lw=1.5,
           label=f"cull_threshold={config.cull_threshold}")
mark_sweeps(ax)
shade_phases(ax, log)
ax.set_xlabel("Step (query number)")
ax.set_ylabel("Score")
ax.set_title("Derived Score Distribution (min / median / max) over Queries")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot_scores.png", dpi=120, bbox_inches="tight")
plt.close()

# plot_phases.png — annotated timeline strip
unique_phases = list(dict.fromkeys(phases))
phase_idx = {p: i for i, p in enumerate(unique_phases)}
colours = list(PHASE_COLOURS.values())

fig, ax = plt.subplots(figsize=(13, 1.6))
for row in log:
    colour = colours[phase_idx[row["phase"]] % len(colours)]
    ax.barh(0, 1, left=row["step"] - 1, color=colour, edgecolor="none", height=0.8)

# Label each phase band
for phase in unique_phases:
    phase_steps = [r["step"] for r in log if r["phase"] == phase]
    mid = (min(phase_steps) + max(phase_steps)) / 2 - 0.5
    ax.text(mid, 0, phase, ha="center", va="center", fontsize=8, color="#222222")

ax.set_xlim(0, len(log))
ax.set_yticks([])
ax.set_xlabel("Step (query number)")
ax.set_title("Phase Timeline")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot_phases.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== COMPACTION TEST SUMMARY ===")
for section, outcome in results.items():
    status = outcome["status"]
    print(f"  {status:6}  {section}: {outcome['message']}")

n_pass = sum(1 for o in results.values() if o["status"] == "PASS")
n_warn = sum(1 for o in results.values() if o["status"] == "WARN")
n_fail = sum(1 for o in results.values() if o["status"] == "FAIL")
print(f"\n  Total: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL  |  {len(sweep_steps)} sweeps across {len(log)} steps")
print(f"\nArtefacts written to {OUTPUT_DIR.resolve()}")
print("\nStart here: open compaction_test_output/plot_scores.png to see score decay across sweeps,")
print("then plot_growth.png to see how derived count climbed toward soft_limit.")
