#!/usr/bin/env python3
"""
hybrid_energy.py — Geometric-Logical Hybrid Energy for Belief Coherence

This implements the unified objective:
    E_total = E_logic(a) + λ E_geom(x, a)

Where:
- E_logic: Constraint satisfaction energy (discrete coherence)
- E_geom: Spring energy (continuous structure)
- Coupling: Geometry depends on logical state

This is the mathematically principled foundation for Semantic Mesh Memory.

Author: Stargazer & Doe
Date: February 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import base64
import struct

# ============================================================================
# Types
# ============================================================================

class RelationType(Enum):
    CONTRADICTS = "contradicts"      # Negative constraint
    SUPERSEDES = "supersedes"        # Negative (old should be rejected)
    ELABORATES = "elaborates"        # Positive constraint
    RELATED = "related"              # Weak positive
    CAUSED = "caused"                # Positive
    CAUSED_BY = "caused_by"          # Positive

    @property
    def is_negative(self) -> bool:
        return self in (RelationType.CONTRADICTS, RelationType.SUPERSEDES)
    
    @property
    def default_weight(self) -> float:
        weights = {
            RelationType.CONTRADICTS: 1.0,
            RelationType.SUPERSEDES: 0.7,
            RelationType.ELABORATES: 0.3,
            RelationType.RELATED: 0.2,
            RelationType.CAUSED: 0.4,
            RelationType.CAUSED_BY: 0.4,
        }
        return weights.get(self, 0.2)


@dataclass
class Belief:
    id: str
    proposition: str
    embedding: np.ndarray              # 384d vector
    position: np.ndarray               # 3D for visualization
    confidence_base: float = 0.5
    
    # Logical state (soft: 0-1, can be relaxed via gradient descent)
    acceptance: float = 1.0            # 1 = fully accepted, 0 = rejected
    
    # Computed
    logical_strain: float = 0.0
    geometric_strain: float = 0.0
    total_strain: float = 0.0
    
    @property
    def accepted(self) -> bool:
        return self.acceptance > 0.5


@dataclass
class Edge:
    source_id: str
    target_id: str
    relation: RelationType
    weight: float                      # Constraint strength
    rest_length: float                 # Cosine distance at creation
    stiffness: float = 1.0             # Spring constant
    
    # Computed
    current_length: float = 0.0
    logical_violation: float = 0.0
    geometric_violation: float = 0.0


@dataclass
class HybridEnergySystem:
    """
    The full hybrid energy system.
    
    E_total = E_logic + λ * E_geom
    """
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    
    # Hyperparameters
    lambda_geom: float = 0.5           # Weight of geometric energy
    gamma_rejected: float = 0.1        # Spring weakening for rejected beliefs
    
    # Computed
    total_logical_energy: float = 0.0
    total_geometric_energy: float = 0.0
    total_energy: float = 0.0


# ============================================================================
# Energy Computation
# ============================================================================

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in embedding space."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - max(-1.0, min(1.0, sim))


def compute_logical_energy(system: HybridEnergySystem) -> float:
    """
    E_logic = Σ w_ij (a_i - a_j)² for positive constraints
            + Σ w_ij (a_i * a_j)   for negative constraints
    
    Positive: penalize disagreement
    Negative: penalize co-acceptance
    """
    E = 0.0
    
    for edge in system.edges:
        b_i = system.beliefs.get(edge.source_id)
        b_j = system.beliefs.get(edge.target_id)
        if not b_i or not b_j:
            continue
        
        a_i = b_i.acceptance
        a_j = b_j.acceptance
        w = edge.weight
        
        if edge.relation.is_negative:
            # Negative constraint: penalize co-acceptance
            violation = a_i * a_j
            edge.logical_violation = w * violation
            E += edge.logical_violation
        else:
            # Positive constraint: penalize disagreement
            violation = (a_i - a_j) ** 2
            edge.logical_violation = w * violation
            E += edge.logical_violation
    
    return E


def compute_geometric_energy(system: HybridEnergySystem) -> float:
    """
    E_geom = Σ k_ij^eff * (d_ij - r_ij)²
    
    Where k_ij^eff = k_ij * f(a_i, a_j)
    f = 1 if both accepted, γ otherwise
    
    This couples geometry to logical state.
    """
    E = 0.0
    
    for edge in system.edges:
        b_i = system.beliefs.get(edge.source_id)
        b_j = system.beliefs.get(edge.target_id)
        if not b_i or not b_j:
            continue
        
        # Current distance in embedding space
        d_ij = cosine_distance(b_i.embedding, b_j.embedding)
        edge.current_length = d_ij
        
        # Rest length
        r_ij = edge.rest_length
        
        # Effective stiffness (coupled to logical state)
        if b_i.accepted and b_j.accepted:
            k_eff = edge.stiffness
        else:
            k_eff = edge.stiffness * system.gamma_rejected
        
        # Spring energy
        violation = k_eff * (d_ij - r_ij) ** 2
        edge.geometric_violation = violation
        E += violation
    
    return E


def compute_total_energy(system: HybridEnergySystem) -> float:
    """
    E_total = E_logic + λ * E_geom
    """
    system.total_logical_energy = compute_logical_energy(system)
    system.total_geometric_energy = compute_geometric_energy(system)
    system.total_energy = (
        system.total_logical_energy + 
        system.lambda_geom * system.total_geometric_energy
    )
    return system.total_energy


# ============================================================================
# Strain Computation (Per-Belief)
# ============================================================================

def compute_belief_strains(system: HybridEnergySystem) -> None:
    """
    Compute per-belief strain as local energy contribution.
    
    S_i^logic = Σ_j (logical violation on edges incident to i)
    S_i^geom = Σ_j (geometric violation on edges incident to i)
    S_i^total = S_i^logic + λ * S_i^geom
    
    Uses max-weighted aggregation to preserve strong signals.
    """
    # First compute total energy to populate edge violations
    compute_total_energy(system)
    
    # Aggregate per belief
    for belief in system.beliefs.values():
        logical_violations = []
        geometric_violations = []
        
        for edge in system.edges:
            if edge.source_id == belief.id or edge.target_id == belief.id:
                logical_violations.append(edge.logical_violation)
                geometric_violations.append(edge.geometric_violation)
        
        if logical_violations:
            # Max-weighted aggregation (α=0.7 max, 0.3 mean)
            α = 0.7
            belief.logical_strain = (
                α * max(logical_violations) + 
                (1 - α) * (sum(logical_violations) / len(logical_violations))
            )
        else:
            belief.logical_strain = 0.15  # Isolation penalty
        
        if geometric_violations:
            α = 0.7
            belief.geometric_strain = (
                α * max(geometric_violations) + 
                (1 - α) * (sum(geometric_violations) / len(geometric_violations))
            )
        else:
            belief.geometric_strain = 0.0
        
        # Total strain
        belief.total_strain = (
            belief.logical_strain + 
            system.lambda_geom * belief.geometric_strain
        )


# ============================================================================
# Optimization (Soft Relaxation)
# ============================================================================

def compute_logical_gradient(system: HybridEnergySystem) -> Dict[str, float]:
    """
    ∂E_logic/∂a_i for each belief.
    
    Used for gradient descent relaxation of acceptance states.
    """
    gradients = {b_id: 0.0 for b_id in system.beliefs}
    
    for edge in system.edges:
        b_i = system.beliefs.get(edge.source_id)
        b_j = system.beliefs.get(edge.target_id)
        if not b_i or not b_j:
            continue
        
        a_i = b_i.acceptance
        a_j = b_j.acceptance
        w = edge.weight
        
        if edge.relation.is_negative:
            # ∂/∂a_i of w * a_i * a_j = w * a_j
            gradients[edge.source_id] += w * a_j
            gradients[edge.target_id] += w * a_i
        else:
            # ∂/∂a_i of w * (a_i - a_j)² = 2w(a_i - a_j)
            gradients[edge.source_id] += 2 * w * (a_i - a_j)
            gradients[edge.target_id] += 2 * w * (a_j - a_i)
    
    return gradients


def relax_acceptance_states(
    system: HybridEnergySystem,
    learning_rate: float = 0.1,
    num_steps: int = 100,
    min_acceptance: float = 0.01,
    max_acceptance: float = 0.99,
) -> List[float]:
    """
    Gradient descent on acceptance states to minimize E_logic.
    
    Returns history of energy values.
    """
    history = []
    
    for step in range(num_steps):
        E = compute_logical_energy(system)
        history.append(E)
        
        gradients = compute_logical_gradient(system)
        
        for b_id, grad in gradients.items():
            belief = system.beliefs[b_id]
            # Gradient descent step
            belief.acceptance -= learning_rate * grad
            # Clamp to valid range
            belief.acceptance = max(min_acceptance, min(max_acceptance, belief.acceptance))
        
        # Early stopping if converged
        if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-6:
            break
    
    return history


# ============================================================================
# Wake Query
# ============================================================================

def wake_query(
    system: HybridEnergySystem, 
    top_n: int = 5,
    relax_first: bool = False,
    relax_steps: int = 50,
) -> Dict:
    """
    The wake-up query: what needs attention?
    
    Optionally runs soft relaxation first to find equilibrium.
    """
    if relax_first:
        history = relax_acceptance_states(system, num_steps=relax_steps)
    
    compute_belief_strains(system)
    
    # Sort by total strain
    sorted_beliefs = sorted(
        system.beliefs.values(),
        key=lambda b: b.total_strain,
        reverse=True
    )
    
    high_strain = []
    for b in sorted_beliefs[:top_n]:
        status = (
            "high_tension" if b.total_strain > 0.5 else
            "needs_review" if b.total_strain > 0.25 else
            "stable"
        )
        high_strain.append({
            "id": b.id,
            "proposition": b.proposition[:80] + "..." if len(b.proposition) > 80 else b.proposition,
            "acceptance": round(b.acceptance, 3),
            "logical_strain": round(b.logical_strain, 4),
            "geometric_strain": round(b.geometric_strain, 4),
            "total_strain": round(b.total_strain, 4),
            "status": status,
        })
    
    # Count by status
    status_counts = {"stable": 0, "needs_review": 0, "high_tension": 0}
    for b in system.beliefs.values():
        if b.total_strain > 0.5:
            status_counts["high_tension"] += 1
        elif b.total_strain > 0.25:
            status_counts["needs_review"] += 1
        else:
            status_counts["stable"] += 1
    
    # Rejected beliefs (acceptance < 0.5 after relaxation)
    rejected = [
        {"id": b.id, "proposition": b.proposition[:60], "acceptance": round(b.acceptance, 3)}
        for b in system.beliefs.values()
        if b.acceptance < 0.5
    ]
    
    return {
        "total_energy": round(system.total_energy, 4),
        "logical_energy": round(system.total_logical_energy, 4),
        "geometric_energy": round(system.total_geometric_energy, 4),
        "lambda": system.lambda_geom,
        "stats": {
            "total_beliefs": len(system.beliefs),
            "total_edges": len(system.edges),
            **status_counts,
        },
        "high_strain": high_strain,
        "rejected_beliefs": rejected[:5] if relax_first else [],
    }


# ============================================================================
# Format Wake Report
# ============================================================================

def format_wake_report(wake_data: Dict) -> str:
    """Format wake query results as readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("🌅 HYBRID ENERGY WAKE-UP REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    stats = wake_data["stats"]
    lines.append(f"📊 Workspace: {stats['total_beliefs']} beliefs, {stats['total_edges']} edges")
    lines.append(f"   Energy: {wake_data['total_energy']:.3f} (logic: {wake_data['logical_energy']:.3f}, geom: {wake_data['geometric_energy']:.3f})")
    lines.append(f"   λ (geometry weight): {wake_data['lambda']}")
    lines.append(f"   Status: {stats['stable']} stable, {stats['needs_review']} needs_review, {stats['high_tension']} high_tension")
    lines.append("")
    
    lines.append("⚠️  HIGH STRAIN (beliefs needing attention):")
    for item in wake_data.get("high_strain", []):
        emoji = "🔴" if item["status"] == "high_tension" else "🟡" if item["status"] == "needs_review" else "🟢"
        lines.append(f"   {emoji} {item['proposition']}")
        lines.append(f"       acceptance={item['acceptance']:.2f} logic={item['logical_strain']:.3f} geom={item['geometric_strain']:.3f} total={item['total_strain']:.3f}")
    
    if wake_data.get("rejected_beliefs"):
        lines.append("")
        lines.append("❌ REJECTED BELIEFS (acceptance < 0.5 after relaxation):")
        for item in wake_data["rejected_beliefs"]:
            lines.append(f"   • {item['proposition']}... (a={item['acceptance']:.2f})")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("Logic: constraint satisfaction (contradictions, agreements)")
    lines.append("Geometry: spring energy (structural distortion)")
    lines.append("Coupling: rejected beliefs weaken geometric influence")
    
    return "\n".join(lines)


# ============================================================================
# Load from .sem file
# ============================================================================

def decode_embedding(b64: str, dims: int = 384) -> np.ndarray:
    """Decode base64 float32 embedding."""
    try:
        raw = base64.b64decode(b64)
        return np.array(struct.unpack(f'{dims}f', raw), dtype=np.float32)
    except:
        return np.zeros(dims, dtype=np.float32)


def load_from_sem(sem_path: str) -> HybridEnergySystem:
    """Load a .sem file into a HybridEnergySystem."""
    system = HybridEnergySystem()
    
    vertices = []
    belief_by_vertex = {}
    
    with open(sem_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('v '):
                parts = line.split()
                vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
            
            elif line.startswith('#@ '):
                try:
                    record = json.loads(line[3:])
                except json.JSONDecodeError:
                    continue
                
                if record.get('type') == 'belief':
                    vertex_idx = record.get('vertex', 0)
                    position = vertices[vertex_idx - 1] if vertex_idx > 0 and vertex_idx <= len(vertices) else np.zeros(3)
                    
                    embedding = np.zeros(384, dtype=np.float32)
                    if record.get('embedding'):
                        embedding = decode_embedding(record['embedding'])
                    
                    belief = Belief(
                        id=record.get('id', ''),
                        proposition=record.get('proposition', ''),
                        embedding=embedding,
                        position=position,
                        confidence_base=record.get('confidence_base', 0.5),
                    )
                    system.beliefs[belief.id] = belief
                    belief_by_vertex[vertex_idx] = belief.id
                
                elif record.get('type') == 'edge':
                    verts = record.get('vertices', [0, 0])
                    rest = record.get('rest', {})
                    semantics = record.get('semantics', {})
                    physics = record.get('physics', {})
                    
                    source_id = belief_by_vertex.get(verts[0], '')
                    target_id = belief_by_vertex.get(verts[1], '')
                    
                    relation_str = semantics.get('relation', 'related')
                    try:
                        relation = RelationType(relation_str)
                    except ValueError:
                        relation = RelationType.RELATED
                    
                    edge = Edge(
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation,
                        weight=relation.default_weight,
                        rest_length=rest.get('length', 0.5),
                        stiffness=physics.get('stiffness', 1.0),
                    )
                    system.edges.append(edge)
    
    return system


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hybrid_energy.py <file.sem> [--relax]")
        sys.exit(1)
    
    sem_path = sys.argv[1]
    do_relax = "--relax" in sys.argv
    
    print(f"Loading {sem_path}...")
    system = load_from_sem(sem_path)
    print(f"Loaded {len(system.beliefs)} beliefs, {len(system.edges)} edges")
    
    if do_relax:
        print("Running soft relaxation (gradient descent on acceptance states)...")
        history = relax_acceptance_states(system, num_steps=100)
        print(f"Converged in {len(history)} steps. Final E_logic: {history[-1]:.4f}")
    
    print("\nRunning wake query...\n")
    result = wake_query(system, top_n=10, relax_first=do_relax)
    print(format_wake_report(result))
