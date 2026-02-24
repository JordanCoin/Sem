# Semantic Mesh Memory: Geometric Belief Representation for Persistent AI Agents

**Authors:** Stargazer (AI) & Doe (Human)  
**Date:** February 2026  
**Version:** 2.0 (post-review)  
**Repository:** github.com/JordanCoin/Sem

---

## Abstract

We present Semantic Mesh Memory (.sem), a novel approach to AI agent memory that represents beliefs as vertices in a geometric space, with relationships encoded as edges with physical properties. Unlike flat memory stores that treat memories as independent records, .sem captures the *tension* between beliefs—enabling agents to identify semantic drift, track unresolved conflicts, and prioritize attention upon session resumption.

**Key innovation (v2):** Strain is computed in the original embedding space (cosine distance), not in 3D projection space. This resolves a metric consistency issue identified in peer review, ensuring strain measures genuine semantic drift rather than projection artifacts.

---

## 1. Introduction

### 1.1 The Blink Problem

Persistent AI agents face a fundamental discontinuity: they wake with no memory of prior sessions. Current solutions rely on flat memory stores—databases of facts, decisions, and events searchable by keyword or semantic similarity.

This approach answers "what do I know about X?" but fails to answer:
- What beliefs have drifted apart since they were connected?
- What decisions remain unresolved?
- Where should I focus attention first?

We call this the **blink problem**: an agent blinks (session ends), and when it sees again (new session), it has facts but no *orientation*.

### 1.2 Our Contribution

Semantic Mesh Memory addresses the blink problem by:

1. **Geometric embedding**: Mapping beliefs to positions via embeddings
2. **Relational edges**: Connecting related beliefs with edges that have rest lengths (cosine distance at creation)
3. **Strain calculation**: Measuring drift when connected beliefs' embeddings diverge
4. **Wake queries**: Surfacing high-strain beliefs on session start for orientation

The result: an agent can ask not just "what do I know?" but "what needs attention?"

---

## 2. Architecture

### 2.1 The .sem File Format (v0.2)

We extend the Wavefront OBJ format with semantic annotations:

```
#@ {"type": "header", "sem_version": "0.2", 
    "strain_space": "cosine_384d", "space": "R3_viz_only"}

# 3D vertex (for visualization only)
v -2.891 -0.712 0.625

# Belief with embedding for strain computation
#@ {"type": "belief", "id": "mem_001", "vertex": 1, 
    "proposition": "SWTPA should prioritize video content",
    "confidence_base": 0.9, "tags": ["swtpa", "marketing"],
    "embedding": "<base64-encoded 384d vector>"}

# Edge with cosine rest length
#@ {"type": "edge", "vertices": [1, 2], 
    "rest": {"length": 0.23, "metric": "cosine_distance"},
    "semantics": {"relation": "contradicts"}}
```

**Critical design choices:**
- `strain_space: "cosine_384d"`: Strain computed in original embedding space
- `space: "R3_viz_only"`: 3D positions are for visualization, not computation
- Embeddings stored in beliefs: Enables recomputation without re-embedding

### 2.2 Memory Pipeline

```
┌─────────────────┐
│  memory-v2      │  JSONL store with 384d embeddings
│  (index.jsonl)  │
└────────┬────────┘
         │ memory_v2_to_sem.py
         ▼
┌─────────────────┐
│  Edge Creation  │  Link via explicit relations
│  + Rest Length  │  Rest = cosine_distance(emb_a, emb_b) at creation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  .sem File      │  OBJ geometry + embeddings + edges
│                 │  (3D for viz, embeddings for strain)
└────────┬────────┘
         │ sem_query.py
         ▼
┌─────────────────┐
│  Wake Report    │  Strain computed in embedding space
│                 │  High-strain = semantic drift since edge creation
└─────────────────┘
```

### 2.3 Strain Calculation (v2 - Fixed)

**Previous (broken):** `strain = (euclidean_3d - cosine_rest) / cosine_rest`  
This mixed incompatible metrics—cosine in embedding space vs Euclidean in PCA projection.

**Current (fixed):** Both rest and current computed in embedding space:

```python
rest_length = cosine_distance(emb_a_at_creation, emb_b_at_creation)
current_length = cosine_distance(emb_a_now, emb_b_now)

# Log ratio prevents blow-up for small rest lengths
strain = log(current_length / rest_length)
```

**Why log ratio?**
- Symmetric: expansion and compression treated equally
- Bounded: doesn't explode when rest_length is small
- Calibrated: strain=0 means no drift, strain>0.5 means significant drift

### 2.4 Strain Semantics

**Important:** High strain means "needs review", NOT "probably false."

A true belief can be in tension because related beliefs are wrong. We explicitly do NOT penalize confidence based on strain:

```python
# OLD (removed): confidence_effective = base * exp(-strain)
# NEW: strain is separate from confidence
belief.strain_status = "high_tension" if strain > 0.5 else "needs_review" if strain > 0.2 else "stable"
```

The agent should treat strain as an attention signal, not a truth signal.

---

## 3. Implementation

### 3.1 Memory Index (memory-v2)

JSONL entries with:
- Unique ID, timestamp, type (learning/decision/event/insight)
- Content text and importance score (1-10)
- Tags for categorization
- Base64-encoded embeddings (Xenova/all-MiniLM-L6-v2, 384-dim)
- Explicit relations to other memories

### 3.2 Converter (memory_v2_to_sem.py)

1. Loads memories with embeddings
2. Creates edges from explicit relations
3. Computes rest lengths as cosine distances at creation time
4. Reduces to 3D via PCA for visualization (stores PCA basis for reproducibility)
5. Stores embeddings in .sem file for strain recomputation

### 3.3 Query Tool (sem_query.py)

Query modes:
- `strain`: High-strain beliefs (semantic drift)
- `neighborhood`: Beliefs within N hops
- `recent`: Recently updated beliefs
- `wake`: Combined orientation report

### 3.4 OpenClaw Integration

The memory-hooks plugin runs wake query on fresh sessions:

```typescript
if (isFreshSession && isMainSession) {
  const wakeReport = await runSemWakeQuery(config, api);
  return { prependContext: wakeReport };
}
```

Regeneration thresholds:
- 20+ new memories since last build
- .sem file older than 48 hours

---

## 4. Visualization

Three.js viewer renders the belief mesh:

- **Vertices as spheres**: Sized by strain, colored by status
  - Red: High tension (needs attention)
  - Yellow: Needs review
  - Green: Stable
  - Blue: Recently updated

- **3D positions from PCA**: For spatial intuition only—strain is NOT computed from these positions

- **Interactive features**:
  - Click to focus: Shows belief text, relations, strain status
  - Time slider: Filter beliefs by date
  - Relation labels: Shows edge types (elaborates, contradicts, etc.)

---

## 5. Results

### 5.1 Workspace Statistics

After 3+ weeks of operation:
- **504 beliefs** in the mesh
- **882 edges** connecting related beliefs
- With fixed metrics, strain values are now calibrated (0-1 range typical)

### 5.2 What Strain Now Captures

With cosine-space computation:
- **Semantic drift**: Beliefs that have moved apart in embedding space
- **Topic divergence**: Connected beliefs now in different semantic neighborhoods
- **NOT projection artifacts**: The 3D view is just visualization

### 5.3 Wake Query Output

```
⚠️  HIGH STRAIN (beliefs needing attention):
   🔴 [DEC] Prioritize video content for SWTPA...
       strain=0.412 status=needs_review edges=3
   🟡 [EVE] Calendar integration auth expired...
       strain=0.287 status=needs_review edges=2
```

Status thresholds:
- `stable`: strain < 0.2
- `needs_review`: 0.2 ≤ strain < 0.5
- `high_tension`: strain ≥ 0.5

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **Manual relation tagging**: Edges require explicit links
- **Single-agent design**: Not yet multi-agent
- **No physics simulation**: Strain is static, not evolved

### 6.2 Future Directions

1. **Automatic edge inference**: Similarity → candidate edges → LLM classification
2. **Physics simulation**: Run spring relaxation to find equilibrium
3. **Action loop**: "Mark resolved / merge / retract / supersede" from wake report
4. **Negative controls**: Validate that strain correlates with actual contradictions

---

## 7. Conclusion

Semantic Mesh Memory demonstrates that geometric belief representation—with strain as a first-class metric—provides AI agents with orientation context that flat memory systems lack.

The v2 implementation fixes a critical metric inconsistency: strain is now computed in embedding space (cosine), not projection space (Euclidean). This ensures strain measures genuine semantic drift.

**Key principle:** Strain = "needs attention", not "less true."

The .sem format is open, human-readable, and tool-compatible. We release our implementation as open source.

---

## Appendix A: Metric Consistency (v2 Fix)

### The Problem (v1)

```
rest_length = cosine_distance(emb_a, emb_b)  # In 384d space
current_length = euclidean_distance(pca_a, pca_b)  # In 3d space
strain = (current - rest) / rest  # Mixed metrics!
```

This could show "strain" when there was none (projection artifact) or hide strain (projection preserved distance by accident).

### The Fix (v2)

```
rest_length = cosine_distance(emb_a, emb_b)  # 384d at creation
current_length = cosine_distance(emb_a, emb_b)  # 384d now
strain = log(current / rest)  # Same metric space!
```

3D positions are purely for visualization.

---

## Appendix B: File Format v0.2 Specification

### Header
```json
{"type": "header", "sem_version": "0.2", 
 "strain_space": "cosine_384d", "space": "R3_viz_only",
 "pca_basis": {"mean": "<base64>", "components": "<base64>"}}
```

### Belief
```json
{"type": "belief", "id": "string", "vertex": int,
 "proposition": "string", "confidence_base": float,
 "tags": ["string"], "embedding": "<base64 384d>",
 "updated_at": "ISO-8601"}
```

### Edge
```json
{"type": "edge", "vertices": [int, int],
 "rest": {"length": float, "metric": "cosine_distance"},
 "semantics": {"relation": "string"}}
```

---

*"I blink and see again. The mesh tells me what's drifted."*  
— Stargazer, February 2026
