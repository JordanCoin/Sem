# Sem — Semantic Mesh Memory

A geometric approach to AI agent memory that represents beliefs as vertices in a mesh, with relationships as edges that can accumulate strain.

**Key insight:** Memory isn't just recall—it's coherence over time. A graph with constraints is a principled way to model that.

## What is .sem?

A `.sem` file is an OBJ-based format with semantic annotations:

```
# 3D vertex (for visualization only)
v -2.891 -0.712 0.625

# Belief with embedding for strain computation
#@ {"type": "belief", "id": "mem_001", "vertex": 1, 
    "proposition": "SWTPA should prioritize video content",
    "embedding": "<base64 384d vector>"}

# Edge with cosine rest length  
#@ {"type": "edge", "vertices": [1, 2],
    "rest": {"length": 0.23, "metric": "cosine_distance"},
    "semantics": {"relation": "contradicts"}}
```

## Why geometric memory?

Flat memory stores answer: "What do I know about X?"

Semantic meshes answer: "What's unresolved? What needs attention?"

By encoding relationships as edges with rest lengths, we can measure **strain**—how much connected beliefs have drifted apart in embedding space.

## v2.0 Changes (Feb 2026)

Fixed metric consistency per reviewer feedback:
- **Strain computed in embedding space** (cosine distance), not 3D projection
- 3D positions are for visualization only
- Strain = `log(current_length / rest_length)` (bounded, symmetric)
- Strain means "needs review", NOT "probably false"

## Quick Start

### Convert memory-v2 to .sem

```bash
python3 converters/memory_v2_to_sem.py input.jsonl output.sem
```

### Query for wake-up context

```bash
python3 tools/sem_query.py output.sem wake --top 5
```

Output:
```
============================================================
🌅 WAKE-UP CONTEXT REPORT
============================================================

📊 Workspace: 504 beliefs, 882 edges
   Embedded: 504 | Metric: cosine_distance_log_ratio
   Status: 500 stable, 3 needs review, 1 high tension

⚠️  HIGH STRAIN (beliefs needing attention):
   🔴 [DEC] Prioritize video content...
       strain=0.412 status=needs_review edges=3
```

### Visualize the mesh

```bash
cd viewer
cp ../generated/your-mesh.sem .
python3 -m http.server 8765
# Open http://localhost:8765
```

## Query Commands

```bash
# High-strain beliefs
python3 tools/sem_query.py mesh.sem strain --top 10

# Neighborhood exploration  
python3 tools/sem_query.py mesh.sem neighborhood <belief_id> --radius 2

# Recent updates
python3 tools/sem_query.py mesh.sem recent --since 2026-02-01

# Wake-up context (combined)
python3 tools/sem_query.py mesh.sem wake --topic "project name" --top 5
```

## File Format v0.2

### Header
```json
{"type": "header", "sem_version": "0.2",
 "strain_space": "cosine_384d", "space": "R3_viz_only"}
```

### Belief
```json
{"type": "belief", "id": "string", "vertex": int,
 "proposition": "string", "confidence_base": float,
 "tags": ["string"], "embedding": "<base64>"}
```

### Edge
```json
{"type": "edge", "vertices": [int, int],
 "rest": {"length": float, "metric": "cosine_distance"},
 "semantics": {"relation": "string"}}
```

## Integration with OpenClaw

The `memory-hooks` plugin runs wake queries on fresh sessions:

```typescript
if (isFreshSession && isMainSession) {
  const wakeReport = await runSemWakeQuery(config, api);
  return { prependContext: wakeReport };
}
```

## Future Work

1. **Automatic edge inference**: Similarity → candidates → LLM classification
2. **Physics simulation**: Spring relaxation to find equilibrium
3. **Action loop**: "Mark resolved / merge / retract" from wake report
4. **Multi-agent meshes**: Shared belief spaces

## Paper

See [paper/semantic-mesh-memory.md](paper/semantic-mesh-memory.md) for the full technical writeup.

---

*"I blink and see again. The mesh tells me what's drifted."*  
— Stargazer, February 2026
