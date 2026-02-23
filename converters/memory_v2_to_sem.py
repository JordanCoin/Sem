#!/usr/bin/env python3
"""
memory_v2_to_sem.py - Convert OpenClaw memory-v2 index to .sem format

This converter maps:
- Memory entries → Beliefs (vertices)
- Relations (caused, related, etc.) → Triangles/Edges
- Importance scores → confidence_base (normalized 0-1)
- Embeddings (384d) → 3D positions via PCA or UMAP

Part of the Sem ecosystem: https://github.com/JordanCoin/Sem
Compatible with: https://github.com/JordanCoin/openclaw-memory-v2

Usage:
    python3 memory_v2_to_sem.py <input.jsonl> <output.sem> [--method pca|umap]
"""

import json
import sys
import base64
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

def decode_base64_embedding(data: str, dims: int = 384) -> Optional[np.ndarray]:
    """Decode base64-encoded float32 embedding."""
    try:
        raw = base64.b64decode(data)
        floats = struct.unpack(f'{dims}f', raw)
        return np.array(floats)
    except Exception as e:
        print(f"Warning: Failed to decode embedding: {e}", file=sys.stderr)
        return None

def reduce_dimensions(embeddings: np.ndarray, method: str = 'pca') -> np.ndarray:
    """Reduce high-dimensional embeddings to 3D for visualization."""
    if embeddings.shape[0] == 0:
        return np.array([])
    
    if method == 'pca':
        # Simple PCA - no sklearn dependency
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Take top 3 components (sorted by eigenvalue descending)
        idx = np.argsort(eigenvalues)[::-1][:3]
        components = eigenvectors[:, idx]
        reduced = centered @ components
        # Normalize to reasonable range
        if reduced.max() > 0:
            reduced = reduced / reduced.max() * 10
        return reduced
    elif method == 'random':
        # Fallback: random projection (fast, preserves some structure)
        np.random.seed(42)
        projection = np.random.randn(embeddings.shape[1], 3)
        projection /= np.linalg.norm(projection, axis=0)
        reduced = embeddings @ projection
        if reduced.max() > 0:
            reduced = reduced / reduced.max() * 10
        return reduced
    else:
        raise ValueError(f"Unknown method: {method}")

def importance_to_confidence(importance: int) -> float:
    """Convert importance (1-10) to confidence_base (0-1)."""
    return max(0.1, min(1.0, importance / 10.0))

def convert_memory_v2_to_sem(input_path: str, output_path: str, method: str = 'pca'):
    """Main conversion function."""
    
    memories = []
    meta = None
    
    # Read memory-v2 JSONL
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get('_meta'):
                meta = obj
            else:
                memories.append(obj)
    
    print(f"Found {len(memories)} memories")
    
    # Extract embeddings for dimension reduction
    embeddings = []
    embedding_indices = []  # Maps embedding index back to memory index
    
    for i, mem in enumerate(memories):
        if 'embedding' in mem and mem['embedding']:
            emb = decode_base64_embedding(mem['embedding'])
            if emb is not None:
                embeddings.append(emb)
                embedding_indices.append(i)
    
    print(f"Found {len(embeddings)} valid embeddings")
    
    # Reduce dimensions
    if embeddings:
        embeddings_array = np.array(embeddings)
        positions_3d = reduce_dimensions(embeddings_array, method)
    else:
        positions_3d = np.array([])
    
    # Build position lookup
    position_lookup = {}
    for idx, mem_idx in enumerate(embedding_indices):
        position_lookup[memories[mem_idx]['id']] = positions_3d[idx]
    
    # Assign default positions for memories without embeddings
    for i, mem in enumerate(memories):
        if mem['id'] not in position_lookup:
            # Place along a line for memories without embeddings
            position_lookup[mem['id']] = np.array([i * 0.5, 0, 0])
    
    # Build relation graph for triangles
    # Relations in memory-v2: caused, caused_by, related, supersedes, contradicts, elaborates
    edges = []
    for mem in memories:
        if 'relations' in mem:
            for rel in mem['relations']:
                # Only add edge if target exists
                target_exists = any(m['id'] == rel['id'] for m in memories)
                if target_exists:
                    edges.append((mem['id'], rel['id'], rel['type']))
    
    # Find triangles (three memories that are all connected)
    # For now, just create edges - triangles require 3-way connections
    triangles = []
    # TODO: Implement triangle detection from relation graph
    
    # Write .sem file
    print(f"Writing {output_path}...")
    with open(output_path, 'w') as f:
        # Header
        header = {
            "type": "header",
            "sem_version": "0.1",
            "space": "R3",
            "units": "arb",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source": {
                "format": "memory-v2",
                "path": str(input_path),
                "embedding_model": meta.get('embeddingModel') if meta else None,
                "reduction_method": method
            }
        }
        f.write(f"#@ {json.dumps(header)}\n\n")
        
        # Build vertex index mapping
        vertex_index = {}  # id -> 1-based vertex index
        
        # Write vertices and beliefs
        for i, mem in enumerate(memories):
            vertex_idx = i + 1  # OBJ is 1-indexed
            vertex_index[mem['id']] = vertex_idx
            
            pos = position_lookup.get(mem['id'], np.array([0, 0, 0]))
            
            # Vertex
            f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            # Belief record
            belief = {
                "type": "belief",
                "id": mem['id'],
                "vertex": vertex_idx,
                "proposition": mem.get('content', ''),
                "confidence_base": importance_to_confidence(mem.get('importance', 5)),
                "tags": mem.get('tags', []),
                "provenance": {
                    "source": "memory-v2",
                    "type": mem.get('type', 'unknown'),
                    "date": mem.get('date'),
                    "context": mem.get('context')
                },
                "updated_at": mem.get('timestamp', datetime.utcnow().isoformat() + "Z")
            }
            f.write(f"#@ {json.dumps(belief)}\n\n")
        
        # Write edges as edge records
        f.write("# Relations as edges\n")
        for i, (src, tgt, rel_type) in enumerate(edges):
            if src in vertex_index and tgt in vertex_index:
                src_idx = vertex_index[src]
                tgt_idx = vertex_index[tgt]
                
                # Fixed baseline rest length = 1.0
                # This creates meaningful strain: 
                # - close beliefs (current < 1.0) = compressed = coherent
                # - distant beliefs (current > 1.0) = stretched = tension
                rest_length = 1.0
                
                edge = {
                    "type": "edge",
                    "id": f"e_{i+1}",
                    "vertices": [src_idx, tgt_idx],
                    "rest": {"length": rest_length},
                    "physics": {
                        "stiffness": 1.0 if rel_type in ['caused', 'caused_by'] else 0.5,
                        "damping": 0.2
                    },
                    "semantics": {
                        "relation": rel_type,
                        "source_id": src,
                        "target_id": tgt
                    },
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                }
                f.write(f"#@ {json.dumps(edge)}\n")
        
        f.write("\n")
        
        # Write triangles (faces) if we found any
        if triangles:
            f.write("# Coherence triangles\n")
            for i, (v1, v2, v3) in enumerate(triangles):
                f.write(f"f {v1} {v2} {v3}\n")
                # Triangle record would go here
        
        # Add a sample request for agents
        sample_request = {
            "type": "request",
            "id": "r_high_strain",
            "name": "Find high-strain regions",
            "input": {
                "select": {
                    "mode": "top_strained",
                    "max_items": 10
                }
            },
            "agent": {
                "interface": "generic",
                "instruction": "Identify beliefs that are in tension and suggest how to reconcile them.",
                "output_format": "patch_v0"
            },
            "apply": {
                "mode": "suggest",
                "target": "workspace"
            }
        }
        f.write(f"\n#@ {json.dumps(sample_request)}\n")
    
    print(f"Done! Wrote {len(memories)} beliefs and {len(edges)} edges")
    print(f"Open with: ./sem_runtime && curl -X POST http://127.0.0.1:7318/v1/files/open -d '{{\"path\":\"{output_path}\"}}'")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'pca'
    
    convert_memory_v2_to_sem(input_path, output_path, method)
