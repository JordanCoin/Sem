# Semantic Mesh Memory

**A geometric approach to AI agent memory where beliefs are vertices, relationships are edges, and strain surfaces what needs attention.**

---

## The Problem

AI agents wake up with no memory. Current solutions give them a database of facts searchable by keyword or similarity.

That answers: *"What do I know about X?"*

It doesn't answer:
- What's unresolved?
- What beliefs have drifted apart?
- Where should I focus first?

We call this the **blink problem**. The agent blinks, and when it sees again, it has facts but no orientation.

---

## The Solution

Represent beliefs as vertices in a geometric space. Connect related beliefs with edges. Give edges a "rest length" — the distance between beliefs when the edge was created.

When beliefs drift apart (embeddings change, new connections form, time passes), edges stretch. That stretch is **strain**.

High strain = "these connected beliefs are pulling apart" = needs attention.

```
          ┌─────────┐
          │ Belief A │ ←── strain = 0.3 ──→ │ Belief B │
          └─────────┘                       └─────────┘
                ↑
                │ strain = 0.7 (high!)
                ↓
          ┌─────────┐
          │ Belief C │
          └─────────┘
```

On wake-up, the agent sees: "Belief A↔C has high strain. Something changed. Look there first."

---

## The Format

`.sem` extends OBJ (3D mesh format) with semantic annotations:

```
# 3D position (visualization only)
v -2.89 -0.71 0.63

# Belief with embedding
#@ {"type": "belief", "id": "mem_001", 
    "proposition": "Video content is primary for SWTPA",
    "embedding": "<base64 384d vector>"}

# Edge with rest length
#@ {"type": "edge", "vertices": [1, 2],
    "rest": {"length": 0.23, "metric": "cosine_distance"},
    "semantics": {"relation": "contradicts"}}
```

**Key design choice:** 3D is for visualization. Strain is computed in embedding space (384d cosine distance). This avoids projection artifacts.

---

## Strain Signals

Strain combines three signals:

| Signal | What it catches | Weight |
|--------|-----------------|--------|
| **Semantic** | Embedding drift (if model changes or content edits) | log(current/rest) |
| **Temporal** | Old belief ↔ new belief connections | age_gap / 30 days × 0.3 |
| **Relational** | Edge type implies tension | contradicts=0.8, supersedes=0.4 |

Isolated beliefs (no edges) get baseline strain — they're orphaned and need linking.

**Critical:** Strain means "needs review", NOT "probably false." A true belief can be strained because its neighbors are wrong.

---

## Auto-Edge Inference

Manual tagging doesn't scale. When a new memory is added:

1. Compute embedding
2. Find top 3 most similar existing memories (cosine > 0.55)
3. Create bidirectional "related" edges
4. Rest length = cosine distance at creation

The mesh grows organically. New memories automatically connect to their semantic neighborhood.

---

## Wake Query

On session start, the agent runs:

```bash
python3 sem_query.py mesh.sem wake --top 5
```

Output:

```
============================================================
🌅 WAKE-UP CONTEXT REPORT
============================================================

📊 Workspace: 507 beliefs, 890 edges
   Status: 503 stable, 3 needs review, 1 high tension

⚠️  HIGH STRAIN (beliefs needing attention):
   🔴 [DEC] Video is primary format...
       strain=0.52 status=high_tension edges=5
   🟡 [EVE] Browser automation pilot...
       strain=0.28 status=needs_review edges=7

============================================================
```

This is the "orientation layer" flat memory lacks.

---

## Visualization

Three.js viewer renders the mesh:

- Vertices as spheres (sized by strain, colored by status)
- Edges as lines (opacity = strain level)
- Click to focus: see belief text + connections
- Time slider: watch mesh evolve

```bash
cd viewer && python3 -m http.server 8765
# Open http://localhost:8765
```

---

## Integration

OpenClaw `memory-hooks` plugin runs wake query on fresh sessions:

```typescript
if (isFreshSession && isMainSession) {
  const wakeReport = await runSemWakeQuery();
  return { prependContext: wakeReport };
}
```

Regeneration thresholds:
- 20+ new memories since last build
- .sem file older than 48 hours

---

## What We Fixed

**v1 (broken):** Mixed cosine distance (384d) with Euclidean distance (3D PCA). Strain measured projection artifacts, not semantic drift.

**v2 (fixed):** Both rest and current computed in embedding space. 3D is visualization only. Strain formula: `log(current/rest)` — bounded, symmetric.

**v2.1 (useful):** Added temporal + relational strain. Now meaningful even with static embeddings.

---

## File Structure

```
Sem/
├── converters/
│   └── memory_v2_to_sem.py    # JSONL → .sem
├── tools/
│   └── sem_query.py           # strain, wake, neighborhood queries
├── viewer/
│   ├── index.html             # Three.js visualization
│   └── serve.sh               # Local server script
├── generated/
│   └── *.sem                   # Generated meshes (gitignored)
└── paper/
    └── semantic-mesh-memory.md # This doc
```

---

## Usage

**Convert memory index to .sem:**
```bash
python3 converters/memory_v2_to_sem.py input.jsonl output.sem
```

**Query for wake context:**
```bash
python3 tools/sem_query.py output.sem wake --top 5
```

**Explore high-strain beliefs:**
```bash
python3 tools/sem_query.py output.sem strain --top 10
```

**Visualize:**
```bash
cd viewer && cp ../generated/mesh.sem . && python3 -m http.server 8765
```

---

## Future

1. **Physics simulation** — Spring relaxation to find equilibrium
2. **LLM relation classification** — Auto-infer "contradicts" vs "elaborates"
3. **Action loop** — "Mark resolved / merge / retract" from wake report
4. **Multi-agent meshes** — Shared belief spaces

---

## The Takeaway

Memory isn't just recall. It's coherence over time.

A graph with constraints — where strain is a first-class metric — gives agents something flat databases can't: a sense of what's unresolved.

*"I blink and see again. The mesh tells me what's drifted."*

---

**Authors:** Stargazer & Doe  
**Date:** February 2026  
**License:** MIT  
**Repo:** github.com/JordanCoin/Sem
