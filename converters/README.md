# Sem Converters

Tools for importing external data formats into .sem files.

## memory_v2_to_sem.py

Converts [OpenClaw memory-v2](https://github.com/JordanCoin/openclaw-memory-v2) index files to .sem format.

### Mapping

| memory-v2 | .sem |
|-----------|------|
| Memory entry | Belief (vertex) |
| `content` | `proposition` |
| `importance` (1-10) | `confidence_base` (0-1) |
| `embedding` (384d) | 3D position via PCA |
| `relations` | Edges with semantic type |
| `type` (decision/event/etc) | `provenance.type` |
| `tags` | `tags` |

### Usage

```bash
python3 memory_v2_to_sem.py <input.jsonl> <output.sem> [--method pca|random]
```

### Example

```bash
# Convert your memory index
python3 converters/memory_v2_to_sem.py ~/.openclaw/workspace/memory/index/memory-index.jsonl generated/my-memory.sem

# Then open with the runtime
./sem_runtime
curl -X POST http://127.0.0.1:7318/v1/files/open -d '{"path":"generated/my-memory.sem"}'
```

### Requirements

- Python 3.8+
- NumPy

### What it produces

- **Beliefs** with 3D positions derived from embedding similarity (PCA)
- **Edges** for each relation (caused, caused_by, related, supersedes, contradicts, elaborates)
- **Stiffness** varies by relation type (causal = 1.0, others = 0.5)
- **Rest lengths** calculated from embedding-derived positions

### Future enhancements

- [ ] Triangle detection from relation graph
- [ ] UMAP for better high-dimensional reduction
- [ ] Bidirectional sync (write .sem changes back to memory-v2)
- [ ] Streaming conversion for large memory indexes
