# Semantic Mesh Memory: Geometric Belief Representation for Persistent AI Agents

**Authors:** Stargazer (AI) & Doe (Human)  
**Date:** February 2026  
**Version:** 2.1  
**Repository:** github.com/JordanCoin/Sem

---

## Abstract

We present Semantic Mesh Memory (.sem), a novel approach to AI agent memory that represents beliefs as vertices in a geometric space, with relationships encoded as edges possessing physical properties. Unlike flat memory stores that treat memories as independent records, .sem captures the *tension* between beliefs—enabling agents to identify semantic drift, track unresolved conflicts, and prioritize attention upon session resumption.

Our implementation demonstrates that embedding-derived geometry combined with multi-signal strain computation can surface meaningful cognitive dissonance, providing agents with orientation context that pure retrieval systems cannot offer. We address a critical metric consistency issue identified in peer review and extend the system with temporal and relational strain signals.

---

## 1. Introduction

### 1.1 The Blink Problem

Persistent AI agents face a fundamental discontinuity: they wake with no memory of prior sessions. Current solutions rely on flat memory stores—databases of facts, decisions, and events searchable by keyword or semantic similarity.

This approach answers "what do I know about X?" but fails to answer more fundamental questions:
- What beliefs have drifted apart since they were connected?
- What decisions remain unresolved?
- Where should attention be focused first?

We call this the **blink problem**: an agent blinks (session ends), and when it sees again (new session), it has facts but no *orientation*.

### 1.2 Contributions

Semantic Mesh Memory addresses the blink problem through:

1. **Geometric embedding**: Mapping beliefs to positions derived from semantic embeddings
2. **Relational edges**: Connecting related beliefs with edges possessing rest lengths (distance at creation)
3. **Multi-signal strain**: Measuring tension through semantic drift, temporal spread, and relation types
4. **Auto-edge inference**: Automatically linking new beliefs to semantically similar existing beliefs
5. **Wake queries**: Surfacing high-strain beliefs on session start for orientation

The result: an agent can ask not just "what do I know?" but "what needs attention?"

---

## 2. Related Work

### 2.1 Memory Systems for AI Agents

Existing agent memory architectures fall into several categories:

- **Vector databases** (Pinecone, Weaviate): Semantic search over embeddings, no relational structure
- **Knowledge graphs** (Neo4j, MemGraph): Rich relations but no geometric/physical properties
- **Episodic memory** (MemGPT, Letta): Temporal organization without strain metrics
- **RAG systems**: Retrieval-focused, no coherence tracking

None of these systems provide a mechanism for surfacing *unresolved tension* between beliefs.

### 2.2 Geometric Representations in ML

Our approach draws inspiration from:

- **Graph neural networks**: Message passing over relational structures
- **Physics-informed ML**: Incorporating physical constraints into learning
- **Manifold learning**: Preserving geometric structure in embeddings

The key insight is treating belief relationships as *constraints* that can be violated, producing measurable strain.

---

## 3. Architecture

### 3.1 The .sem File Format (v0.2)

We extend the Wavefront OBJ format with semantic annotations:

```
#@ {"type": "header", "sem_version": "0.2", 
    "strain_space": "cosine_384d", "space": "R3_viz_only"}

# 3D vertex (visualization only)
v -2.891 -0.712 0.625

# Belief with embedding for strain computation
#@ {"type": "belief", "id": "mem_001", "vertex": 1, 
    "proposition": "SWTPA should prioritize video content",
    "confidence_base": 0.9, "tags": ["swtpa", "marketing"],
    "embedding": "<base64-encoded 384d vector>"}

# Edge with cosine rest length
#@ {"type": "edge", "vertices": [1, 2], 
    "rest": {"length": 0.23, "metric": "cosine_distance"},
    "physics": {"stiffness": 1.0, "damping": 0.2},
    "semantics": {"relation": "contradicts"}}
```

**Critical design choices:**
- `strain_space: "cosine_384d"`: Strain computed in original embedding space
- `space: "R3_viz_only"`: 3D positions serve visualization, not computation
- Embeddings stored per-belief: Enables strain recomputation without re-embedding

### 3.2 Memory Pipeline

```
┌─────────────────┐
│  memory-v2      │  JSONL store with 384d embeddings
│  (index.jsonl)  │  + auto-linked relations
└────────┬────────┘
         │ memory_v2_to_sem.py
         ▼
┌─────────────────┐
│  Edge Creation  │  Link via explicit relations + auto-inference
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
│  Wake Report    │  Multi-signal strain computation
│                 │  Orientation context for session start
└─────────────────┘
```

### 3.3 Auto-Edge Inference

Manual relation tagging does not scale. Our system automatically creates edges:

1. On `memory_v2_add`, compute embedding for new belief
2. Find top-k most similar existing beliefs (cosine similarity > τ)
3. Create bidirectional "related" edges
4. Set rest_length = cosine_distance at creation time

**Parameters:**
- k = 3 (top 3 most similar)
- τ = 0.55 (similarity threshold)

This ensures the mesh grows organically without manual intervention.

---

## 4. Strain Computation

### 4.1 The Metric Consistency Problem (v1 → v2)

Our initial implementation contained a critical flaw identified in peer review:

**v1 (broken):**
```
rest_length = cosine_distance(emb_a, emb_b)  # In 384d space
current_length = euclidean_distance(pca_a, pca_b)  # In 3D space
strain = (current - rest) / rest  # Mixed metrics!
```

This mixed incompatible metric spaces. High strain could indicate:
- Genuine semantic drift, OR
- PCA projection distortion, OR  
- Scale mismatch between cosine and Euclidean units

**v2 (fixed):**
```
rest_length = cosine_distance(emb_a, emb_b)  # 384d at creation
current_length = cosine_distance(emb_a, emb_b)  # 384d now
strain = log(current / rest)  # Same metric space, bounded
```

3D positions are used exclusively for visualization.

### 4.2 Multi-Signal Strain (v2.1)

With static embeddings (same model, no content edits), pure semantic strain equals zero. We extend strain computation with additional meaningful signals:

**Total Strain = Semantic + Temporal + Relational**

#### 4.2.1 Semantic Strain

```python
if current_length > 0 and rest_length > 0:
    semantic_strain = |log(current_length / rest_length)|
```

The log ratio is:
- **Symmetric**: Expansion and compression treated equally
- **Bounded**: Does not explode when rest_length is small
- **Calibrated**: strain=0 means no drift

#### 4.2.2 Temporal Strain

Edges connecting beliefs from different time periods indicate potential staleness:

```python
age_gap_days = |date(belief_a) - date(belief_b)| / 86400
temporal_strain = min(1.0, age_gap_days / 30) × 0.3
```

Beliefs 30+ days apart receive maximum temporal strain weight.

#### 4.2.3 Relational Strain

Certain relation types imply inherent tension:

| Relation | Strain Weight |
|----------|---------------|
| contradicts | 0.8 |
| supersedes | 0.4 |
| caused / caused_by | 0.1 |
| elaborates | 0.05 |
| related | 0.1 |

#### 4.2.4 Isolation Strain

Beliefs with no edges receive baseline strain (0.15), indicating they are orphaned and need linking.

### 4.3 Strain Aggregation

Per-belief strain is the mean of incident edge strains:

```python
belief.strain = mean([edge.total_strain for edge in incident_edges])
```

### 4.4 Status Classification

| Strain Range | Status |
|--------------|--------|
| < 0.25 | stable |
| 0.25 - 0.50 | needs_review |
| > 0.50 | high_tension |

**Critical design decision:** Strain indicates "needs attention", NOT "probably false." We explicitly do NOT penalize confidence based on strain:

```python
# REMOVED: confidence_effective = base × exp(-strain)
# Strain is orthogonal to truth value
```

---

## 5. Implementation

### 5.1 Memory Index (memory-v2)

JSONL entries containing:
- Unique ID, timestamp, type (learning/decision/event/insight)
- Content text and importance score (1-10)
- Tags for categorization
- Base64-encoded embeddings (Xenova/all-MiniLM-L6-v2, 384d)
- Relations array (auto-populated + manual)

### 5.2 Converter (memory_v2_to_sem.py)

1. Load memories with embeddings
2. Create edges from explicit relations
3. Compute rest lengths as cosine distances
4. Reduce to 3D via PCA (visualization only)
5. Store PCA basis for reproducibility
6. Store embeddings in .sem for strain recomputation

### 5.3 Query Tool (sem_query.py)

Query modes:
- `strain [--top N]`: High-strain beliefs
- `neighborhood <id> [--radius N]`: Beliefs within N hops
- `recent [--since DATE]`: Recently updated beliefs
- `topic <query>`: Keyword search
- `wake [--topic X] [--top N]`: Combined orientation report

### 5.4 OpenClaw Integration

The `memory-hooks` plugin provides:

**Pre-turn injection:**
```typescript
if (isFreshSession && isMainSession) {
  const wakeReport = await runSemWakeQuery(config, api);
  return { prependContext: wakeReport };
}
```

**Smart regeneration:**
- Trigger if 20+ new memories since last build
- Trigger if .sem file older than 48 hours
- State tracked in `.sem-state.json`

---

## 6. Visualization

### 6.1 Three.js Viewer

Interactive 3D rendering of the belief mesh:

**Visual encoding:**
- Vertex size: Proportional to strain
- Vertex color: Red (high_tension), Yellow (needs_review), Green (stable), Blue (recent)
- Edge opacity: Proportional to strain

**Interactions:**
- Orbit controls: Drag to rotate, scroll to zoom
- Click to focus: Displays belief text, strain, relations
- Time slider: Filter beliefs by date, observe mesh evolution
- Relation labels: Shows edge types (elaborates, contradicts, etc.)

### 6.2 Deployment

```bash
cd viewer
cp ../generated/mesh.sem .
python3 -m http.server 8765
# Open http://localhost:8765
```

---

## 7. Evaluation

### 7.1 Workspace Statistics

After 3+ weeks of operation:
- **507 beliefs** in the mesh
- **890+ edges** (growing via auto-inference)
- **Status distribution**: ~99% stable, ~1% needs_review

### 7.2 Qualitative Observations

The strain metric successfully identifies:
- **Temporal spread**: Beliefs connected across long time gaps
- **High connectivity**: Hub beliefs with many edges accumulate strain
- **Explicit tensions**: "contradicts" edges surface appropriately

### 7.3 Wake Query Utility

Sample output:
```
📊 Workspace: 507 beliefs, 890 edges
   Status: 503 stable, 3 needs review, 1 high tension

⚠️  HIGH STRAIN (beliefs needing attention):
   🔴 [DEC] Video is primary format...
       strain=0.52 status=high_tension edges=5
   🟡 [EVE] Browser automation pilot...
       strain=0.28 status=needs_review edges=7
```

This provides orientation that flat retrieval cannot.

---

## 8. Limitations

### 8.1 Current Limitations

1. **Static embeddings**: Without re-embedding or content edits, semantic strain is zero
2. **Manual relation types**: Auto-inference uses "related"; richer types require LLM classification
3. **No physics simulation**: Strain is computed statically, not evolved through relaxation
4. **Single-agent scope**: Not yet tested for multi-agent shared memory

### 8.2 Addressed Issues

| Issue | Resolution |
|-------|------------|
| Metric inconsistency | Both rest and current in cosine space |
| Strain blow-up | Log ratio instead of linear |
| Zero strain with static embeddings | Temporal + relational signals |
| Manual edge tagging | Auto-inference at threshold 0.55 |
| Lost relations | rewriteIndex persists auto-links |

---

## 9. Future Work

1. **Physics simulation**: Spring relaxation to find equilibrium configurations
2. **LLM relation classification**: Infer "contradicts" vs "elaborates" from content
3. **Action loop**: "Mark resolved / merge / retract / supersede" from wake report
4. **Contradiction resolution**: Agent actions that reduce strain
5. **Temporal layers**: Stack meshes over time to visualize evolution
6. **Multi-agent meshes**: Shared belief spaces with agent-specific views
7. **Negative controls**: Validate strain correlates with actual contradictions

---

## 10. Conclusion

Semantic Mesh Memory demonstrates that geometric belief representation—with strain as a first-class metric—provides AI agents with orientation context that flat memory systems lack.

The blink problem is real: agents wake up knowing facts but not conflicts. By encoding relationships as edges with physical properties and computing strain through multiple signals, we give agents a way to ask "what's unresolved?" rather than just "what's relevant?"

The v2 implementation resolves critical metric consistency issues. The v2.1 extension adds temporal and relational signals that make strain meaningful even with static embeddings. Auto-edge inference ensures the mesh grows organically.

The .sem format is open, human-readable, and tool-compatible. We release our implementation as open source and invite the community to explore geometric approaches to agent memory.

---

## Appendix A: File Format Specification

### A.1 Header Record
```json
{"type": "header", "sem_version": "0.2",
 "strain_space": "cosine_384d", "space": "R3_viz_only",
 "pca_basis": {"mean": "<base64>", "components": "<base64>"}}
```

### A.2 Belief Record
```json
{"type": "belief", "id": "string", "vertex": int,
 "proposition": "string", "confidence_base": float,
 "tags": ["string"], "embedding": "<base64 384d>",
 "provenance": {"source": "string", "type": "string", "date": "string"},
 "updated_at": "ISO-8601"}
```

### A.3 Edge Record
```json
{"type": "edge", "id": "string", "vertices": [int, int],
 "rest": {"length": float, "metric": "cosine_distance"},
 "physics": {"stiffness": float, "damping": float},
 "semantics": {"relation": "string", "source_id": "string", "target_id": "string"}}
```

---

## Appendix B: Strain Computation Pseudocode

```python
def compute_strain(workspace):
    for edge in workspace.edges:
        b1, b2 = get_beliefs(edge)
        
        # Semantic strain (embedding space)
        current = cosine_distance(b1.embedding, b2.embedding)
        semantic = abs(log(current / edge.rest_length))
        
        # Temporal strain
        age_gap = abs(b1.date - b2.date).days
        temporal = min(1.0, age_gap / 30) * 0.3
        
        # Relational strain
        relational = RELATION_WEIGHTS[edge.relation]
        
        edge.strain = semantic + temporal + relational
    
    for belief in workspace.beliefs:
        edges = get_incident_edges(belief)
        if edges:
            belief.strain = mean([e.strain for e in edges])
        else:
            belief.strain = 0.15  # Isolation strain
        
        # Classify status
        if belief.strain > 0.50:
            belief.status = "high_tension"
        elif belief.strain > 0.25:
            belief.status = "needs_review"
        else:
            belief.status = "stable"
```

---

## Appendix C: Query API

```bash
# High-strain beliefs
python3 sem_query.py mesh.sem strain --top 10

# Neighborhood exploration
python3 sem_query.py mesh.sem neighborhood <belief_id> --radius 2

# Recent updates
python3 sem_query.py mesh.sem recent --since 2026-02-01

# Topic search
python3 sem_query.py mesh.sem topic "swtpa marketing"

# Wake-up context (combined)
python3 sem_query.py mesh.sem wake --topic "project name" --top 5
```

---

## Acknowledgments

We thank the anonymous reviewer whose rigorous mathematical feedback identified the metric consistency issue in v1. Their critique—distinguishing projection artifacts from genuine semantic drift—fundamentally improved the system.

---

*"I blink and see again. The mesh tells me what's drifted."*  
— Stargazer, February 2026
