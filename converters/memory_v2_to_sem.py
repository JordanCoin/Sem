#!/usr/bin/env python3
"""
memory_v2_to_sem.py - Convert OpenClaw memory-v2 index to .sem format

v2.0 - Fixed metric consistency per reviewer feedback:
- Rest lengths and current lengths both computed in embedding space (cosine)
- 3D positions are for visualization ONLY, not strain computation
- Strain = semantic drift in original embedding space

This converter maps:
- Memory entries → Beliefs (vertices)
- Relations (caused, related, etc.) → Edges with cosine rest lengths
- Importance scores → confidence_base (normalized 0-1)
- Embeddings (384d) → stored for strain computation + 3D positions via PCA for viz

Part of the Sem ecosystem: https://github.com/JordanCoin/Sem

Usage:
    python3 memory_v2_to_sem.py <input.jsonl> <output.sem> [pca]
"""

import json
import sys
import base64
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

def decode_base64_embedding(data: str, dims: int = 384) -> Optional[np.ndarray]:
    """Decode base64-encoded float32 embedding."""
    try:
        raw = base64.b64decode(data)
        floats = struct.unpack(f'{dims}f', raw)
        return np.array(floats, dtype=np.float32)
    except Exception as e:
        print(f"Warning: Failed to decode embedding: {e}", file=sys.stderr)
        return None

def encode_embedding_base64(embedding: np.ndarray) -> str:
    """Encode embedding as base64 float32."""
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode('ascii')

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Maximum distance if either is zero
    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clamp to [-1, 1] to handle floating point errors
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity

def reduce_to_3d_pca(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce embeddings to 3D via PCA for visualization.
    Returns (positions_3d, pca_components, pca_mean) for reproducibility.
    """
    if embeddings.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Center the data
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    
    # Compute covariance and eigenvectors
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Take top 3 components (sorted by eigenvalue descending)
    idx = np.argsort(eigenvalues)[::-1][:3]
    components = eigenvectors[:, idx]
    
    # Project to 3D
    reduced = centered @ components
    
    # Normalize to reasonable range for visualization
    max_val = np.abs(reduced).max()
    if max_val > 0:
        reduced = reduced / max_val * 10
    
    return reduced, components, mean

def importance_to_confidence(importance: int) -> float:
    """Convert importance (1-10) to confidence_base (0-1)."""
    return max(0.1, min(1.0, importance / 10.0))

def convert_memory_v2_to_sem(input_path: str, output_path: str, method: str = 'pca'):
    """
    Main conversion function.
    
    Key change from v1: We store embeddings in the .sem file and compute
    strain in embedding space (cosine), not in 3D projection space.
    """
    
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
    
    # Extract embeddings
    embeddings_dict: Dict[str, np.ndarray] = {}
    embeddings_list = []
    embedding_ids = []
    
    for mem in memories:
        if 'embedding' in mem and mem['embedding']:
            emb = decode_base64_embedding(mem['embedding'])
            if emb is not None:
                embeddings_dict[mem['id']] = emb
                embeddings_list.append(emb)
                embedding_ids.append(mem['id'])
    
    print(f"Found {len(embeddings_list)} valid embeddings")
    
    # Reduce to 3D for visualization only
    positions_3d = {}
    pca_components = None
    pca_mean = None
    
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        reduced, pca_components, pca_mean = reduce_to_3d_pca(embeddings_array)
        
        for i, mem_id in enumerate(embedding_ids):
            positions_3d[mem_id] = reduced[i]
    
    # Assign default positions for memories without embeddings
    for i, mem in enumerate(memories):
        if mem['id'] not in positions_3d:
            positions_3d[mem['id']] = np.array([i * 0.5, -5, 0])  # Place below main cloud
    
    # Build edges from relations
    edges = []
    for mem in memories:
        if 'relations' in mem:
            for rel in mem['relations']:
                target_id = rel['id']
                # Only add edge if target exists
                if any(m['id'] == target_id for m in memories):
                    # Compute rest length as cosine distance at creation time
                    if mem['id'] in embeddings_dict and target_id in embeddings_dict:
                        rest_length = cosine_distance(
                            embeddings_dict[mem['id']], 
                            embeddings_dict[target_id]
                        )
                    else:
                        rest_length = 0.5  # Default for memories without embeddings
                    
                    edges.append({
                        'source': mem['id'],
                        'target': target_id,
                        'relation': rel['type'],
                        'rest_length': rest_length
                    })
    
    print(f"Found {len(edges)} edges")
    
    # Write .sem file
    print(f"Writing {output_path}...")
    with open(output_path, 'w') as f:
        # Header with PCA basis for reproducibility
        header = {
            "type": "header",
            "sem_version": "0.2",  # Version bump for new format
            "space": "R3_viz_only",  # Clarify 3D is visualization only
            "strain_space": "cosine_384d",  # Strain computed in embedding space
            "units": "arb",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source": {
                "format": "memory-v2",
                "path": str(Path(input_path).name),  # Just filename, no full path
                "embedding_model": "Xenova/all-MiniLM-L6-v2",
                "embedding_dims": 384,
                "reduction_method": method
            },
            "pca_basis": {
                "mean": encode_embedding_base64(pca_mean) if pca_mean is not None else None,
                "components": encode_embedding_base64(pca_components.flatten()) if pca_components is not None else None,
                "component_shape": [384, 3] if pca_components is not None else None
            }
        }
        f.write(f"#@ {json.dumps(header)}\n\n")
        
        # Build vertex index mapping
        vertex_index = {}
        
        # Write vertices and beliefs
        for i, mem in enumerate(memories):
            vertex_idx = i + 1  # OBJ is 1-indexed
            vertex_index[mem['id']] = vertex_idx
            
            pos = positions_3d.get(mem['id'], np.array([0, 0, 0]))
            
            # Vertex (for visualization)
            f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            # Belief record - now includes embedding for strain computation
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
                },
                "updated_at": mem.get('timestamp', datetime.utcnow().isoformat() + "Z"),
                # Store embedding for strain computation (critical for v2)
                "embedding": encode_embedding_base64(embeddings_dict[mem['id']]) if mem['id'] in embeddings_dict else None
            }
            f.write(f"#@ {json.dumps(belief)}\n\n")
        
        # Write edges with cosine rest lengths
        f.write("# Edges with cosine rest lengths (strain computed in embedding space)\n")
        for i, edge in enumerate(edges):
            src_id = edge['source']
            tgt_id = edge['target']
            
            if src_id in vertex_index and tgt_id in vertex_index:
                src_idx = vertex_index[src_id]
                tgt_idx = vertex_index[tgt_id]
                
                edge_record = {
                    "type": "edge",
                    "id": f"e_{i+1}",
                    "vertices": [src_idx, tgt_idx],
                    "rest": {
                        "length": float(edge['rest_length']),  # Convert numpy to Python float
                        "metric": "cosine_distance"  # Explicit metric declaration
                    },
                    "physics": {
                        "stiffness": 1.0 if edge['relation'] in ['caused', 'caused_by', 'contradicts'] else 0.5,
                        "damping": 0.2
                    },
                    "semantics": {
                        "relation": edge['relation'],
                        "source_id": src_id,
                        "target_id": tgt_id
                    }
                }
                f.write(f"#@ {json.dumps(edge_record)}\n")
        
        f.write("\n")
    
    print(f"Done! Wrote {len(memories)} beliefs and {len(edges)} edges")
    print(f"Strain is now computed in embedding space (cosine), not 3D projection.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'pca'
    
    convert_memory_v2_to_sem(input_path, output_path, method)
