#!/usr/bin/env python3
"""
sem_query.py - Query a .sem file for strain, neighborhoods, and wake-up context

This is the "blink recovery" tool - helps an agent understand what needs attention
after waking up from a session break.

Usage:
    python3 sem_query.py <file.sem> strain [--top N]
    python3 sem_query.py <file.sem> neighborhood <belief_id> [--radius N]
    python3 sem_query.py <file.sem> recent [--since ISO_DATE]
    python3 sem_query.py <file.sem> wake [--topic "last topic"] [--top N]

Part of: https://github.com/JordanCoin/Sem
"""

import json
import sys
import re
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Belief:
    id: str
    vertex: int
    position: Tuple[float, float, float]
    proposition: str
    confidence_base: float
    tags: List[str]
    provenance: Dict[str, Any]
    updated_at: str
    
    # Computed
    strain: float = 0.0
    confidence_effective: float = 0.0

@dataclass
class Edge:
    id: str
    vertices: Tuple[int, int]
    rest_length: float
    current_length: float = 0.0
    stiffness: float = 1.0
    damping: float = 0.2
    relation: str = "related"
    source_id: str = ""
    target_id: str = ""
    
    @property
    def strain(self) -> float:
        """Normalized strain: (current - rest) / rest"""
        if self.rest_length <= 0:
            return 0.0
        return (self.current_length - self.rest_length) / self.rest_length

@dataclass 
class SemWorkspace:
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    vertex_to_belief: Dict[int, str] = field(default_factory=dict)
    belief_edges: Dict[str, List[Edge]] = field(default_factory=lambda: defaultdict(list))
    
    def calculate_strain(self):
        """Calculate current edge lengths and belief strain from accumulated edges."""
        # For each belief, accumulate strain from incident edges
        strain_sums = defaultdict(float)
        strain_counts = defaultdict(int)
        
        for edge in self.edges:
            # Get positions
            b1_id = self.vertex_to_belief.get(edge.vertices[0])
            b2_id = self.vertex_to_belief.get(edge.vertices[1])
            
            if not b1_id or not b2_id:
                continue
                
            b1 = self.beliefs.get(b1_id)
            b2 = self.beliefs.get(b2_id)
            
            if not b1 or not b2:
                continue
            
            # Calculate current length
            dx = b1.position[0] - b2.position[0]
            dy = b1.position[1] - b2.position[1]
            dz = b1.position[2] - b2.position[2]
            edge.current_length = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Accumulate strain for both endpoints
            s = abs(edge.strain)
            strain_sums[b1_id] += s
            strain_counts[b1_id] += 1
            strain_sums[b2_id] += s
            strain_counts[b2_id] += 1
        
        # Average strain per belief and compute effective confidence
        alpha = 1.0  # dampening factor
        for belief_id, belief in self.beliefs.items():
            if strain_counts[belief_id] > 0:
                belief.strain = strain_sums[belief_id] / strain_counts[belief_id]
            else:
                belief.strain = 0.0
            
            # confidence_effective = confidence_base * exp(-alpha * strain)
            belief.confidence_effective = belief.confidence_base * math.exp(-alpha * belief.strain)

def parse_sem_file(path: str) -> SemWorkspace:
    """Parse a .sem file into a queryable workspace."""
    workspace = SemWorkspace()
    
    vertices = []  # List of (x, y, z) - 0-indexed internally
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # OBJ vertex
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
            
            # Semantic record
            elif line.startswith('#@ '):
                json_str = line[3:]
                try:
                    record = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                
                record_type = record.get('type')
                
                if record_type == 'belief':
                    vertex_idx = record.get('vertex', 0)
                    # Vertex index in OBJ is 1-based, our list is 0-based
                    pos = vertices[vertex_idx - 1] if vertex_idx > 0 and vertex_idx <= len(vertices) else (0, 0, 0)
                    
                    belief = Belief(
                        id=record.get('id', ''),
                        vertex=vertex_idx,
                        position=pos,
                        proposition=record.get('proposition', ''),
                        confidence_base=record.get('confidence_base', 0.5),
                        tags=record.get('tags', []),
                        provenance=record.get('provenance', {}),
                        updated_at=record.get('updated_at', '')
                    )
                    workspace.beliefs[belief.id] = belief
                    workspace.vertex_to_belief[vertex_idx] = belief.id
                
                elif record_type == 'edge':
                    verts = record.get('vertices', [0, 0])
                    rest = record.get('rest', {})
                    physics = record.get('physics', {})
                    semantics = record.get('semantics', {})
                    
                    edge = Edge(
                        id=record.get('id', ''),
                        vertices=(verts[0], verts[1]),
                        rest_length=rest.get('length', 1.0),
                        stiffness=physics.get('stiffness', 1.0),
                        damping=physics.get('damping', 0.2),
                        relation=semantics.get('relation', 'related'),
                        source_id=semantics.get('source_id', ''),
                        target_id=semantics.get('target_id', '')
                    )
                    workspace.edges.append(edge)
                    
                    # Track edges per belief
                    if edge.source_id:
                        workspace.belief_edges[edge.source_id].append(edge)
                    if edge.target_id:
                        workspace.belief_edges[edge.target_id].append(edge)
    
    # Calculate strain
    workspace.calculate_strain()
    
    return workspace

def query_high_strain(workspace: SemWorkspace, top_n: int = 10) -> List[Dict]:
    """Return beliefs with highest strain (most tension)."""
    sorted_beliefs = sorted(
        workspace.beliefs.values(),
        key=lambda b: b.strain,
        reverse=True
    )[:top_n]
    
    results = []
    for b in sorted_beliefs:
        results.append({
            'id': b.id,
            'proposition': b.proposition,
            'strain': round(b.strain, 4),
            'confidence_base': b.confidence_base,
            'confidence_effective': round(b.confidence_effective, 4),
            'tags': b.tags,
            'type': b.provenance.get('type', 'unknown')
        })
    
    return results

def query_neighborhood(workspace: SemWorkspace, center_id: str, radius: int = 2) -> Dict:
    """Return beliefs within N hops of a center belief."""
    if center_id not in workspace.beliefs:
        return {'error': f'Belief {center_id} not found'}
    
    visited = {center_id}
    frontier = [center_id]
    
    for _ in range(radius):
        next_frontier = []
        for belief_id in frontier:
            for edge in workspace.belief_edges.get(belief_id, []):
                neighbor_id = edge.target_id if edge.source_id == belief_id else edge.source_id
                if neighbor_id and neighbor_id not in visited and neighbor_id in workspace.beliefs:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        frontier = next_frontier
    
    neighbors = []
    for belief_id in visited:
        b = workspace.beliefs[belief_id]
        neighbors.append({
            'id': b.id,
            'proposition': b.proposition,
            'strain': round(b.strain, 4),
            'confidence_effective': round(b.confidence_effective, 4),
            'is_center': belief_id == center_id
        })
    
    return {
        'center': center_id,
        'radius': radius,
        'count': len(neighbors),
        'beliefs': neighbors
    }

def query_recent(workspace: SemWorkspace, since: Optional[str] = None, top_n: int = 20) -> List[Dict]:
    """Return recently updated beliefs."""
    beliefs_with_dates = []
    
    for b in workspace.beliefs.values():
        try:
            # Parse ISO date
            dt_str = b.updated_at.replace('Z', '+00:00')
            dt = datetime.fromisoformat(dt_str)
            beliefs_with_dates.append((dt, b))
        except:
            continue
    
    # Sort by date descending
    beliefs_with_dates.sort(key=lambda x: x[0], reverse=True)
    
    # Filter by since date if provided
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            beliefs_with_dates = [(dt, b) for dt, b in beliefs_with_dates if dt >= since_dt]
        except:
            pass
    
    results = []
    for dt, b in beliefs_with_dates[:top_n]:
        results.append({
            'id': b.id,
            'proposition': b.proposition,
            'updated_at': b.updated_at,
            'strain': round(b.strain, 4),
            'type': b.provenance.get('type', 'unknown')
        })
    
    return results

def query_by_topic(workspace: SemWorkspace, topic: str, top_n: int = 10) -> List[Dict]:
    """Find beliefs matching a topic (keyword search in proposition/tags)."""
    topic_lower = topic.lower()
    keywords = topic_lower.split()
    
    scored = []
    for b in workspace.beliefs.values():
        prop_lower = b.proposition.lower()
        tags_lower = ' '.join(b.tags).lower()
        
        score = 0
        for kw in keywords:
            if kw in prop_lower:
                score += 2
            if kw in tags_lower:
                score += 1
        
        if score > 0:
            scored.append((score, b))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, b in scored[:top_n]:
        results.append({
            'id': b.id,
            'proposition': b.proposition,
            'match_score': score,
            'strain': round(b.strain, 4),
            'tags': b.tags
        })
    
    return results

def wake_query(workspace: SemWorkspace, topic: Optional[str] = None, top_n: int = 5) -> Dict:
    """
    The "blink recovery" query - what an agent should see on wake-up.
    
    Returns:
    - high_strain: beliefs that need reconciling (contradictions/tension)
    - topic_neighborhood: if topic provided, relevant context
    - recent_changes: what was updated recently
    - stats: overall workspace health
    """
    result = {
        'wake_time': datetime.now(timezone.utc).isoformat(),
        'stats': {
            'total_beliefs': len(workspace.beliefs),
            'total_edges': len(workspace.edges),
            'avg_strain': round(sum(b.strain for b in workspace.beliefs.values()) / max(len(workspace.beliefs), 1), 4)
        },
        'high_strain': query_high_strain(workspace, top_n),
        'recent': query_recent(workspace, top_n=top_n)
    }
    
    if topic:
        topic_matches = query_by_topic(workspace, topic, top_n)
        result['topic_context'] = topic_matches
        
        # If we found a match, also get its neighborhood
        if topic_matches:
            best_match_id = topic_matches[0]['id']
            result['topic_neighborhood'] = query_neighborhood(workspace, best_match_id, radius=2)
    
    return result

def format_wake_report(wake_data: Dict) -> str:
    """Format wake query results as readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("🌅 WAKE-UP CONTEXT REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    stats = wake_data['stats']
    lines.append(f"📊 Workspace: {stats['total_beliefs']} beliefs, {stats['total_edges']} edges")
    lines.append(f"   Average strain: {stats['avg_strain']}")
    lines.append("")
    
    lines.append("⚠️  HIGH STRAIN (beliefs in tension):")
    for item in wake_data.get('high_strain', [])[:5]:
        strain_bar = "█" * int(item['strain'] * 10) if item['strain'] > 0 else "░"
        lines.append(f"   [{item['type'][:3].upper()}] {item['proposition'][:60]}...")
        lines.append(f"       strain={item['strain']} conf={item['confidence_effective']} tags={item['tags'][:3]}")
    lines.append("")
    
    if 'topic_context' in wake_data and wake_data['topic_context']:
        lines.append("🎯 TOPIC CONTEXT:")
        for item in wake_data['topic_context'][:3]:
            lines.append(f"   • {item['proposition'][:70]}...")
    lines.append("")
    
    lines.append("🕐 RECENT UPDATES:")
    for item in wake_data.get('recent', [])[:5]:
        lines.append(f"   [{item['type'][:3].upper()}] {item['proposition'][:60]}...")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    sem_path = sys.argv[1]
    command = sys.argv[2]
    
    workspace = parse_sem_file(sem_path)
    
    if command == 'strain':
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        results = query_high_strain(workspace, top_n)
        print(json.dumps(results, indent=2))
    
    elif command == 'neighborhood':
        belief_id = sys.argv[3] if len(sys.argv) > 3 else ''
        radius = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        results = query_neighborhood(workspace, belief_id, radius)
        print(json.dumps(results, indent=2))
    
    elif command == 'recent':
        since = sys.argv[3] if len(sys.argv) > 3 else None
        results = query_recent(workspace, since)
        print(json.dumps(results, indent=2))
    
    elif command == 'topic':
        topic = sys.argv[3] if len(sys.argv) > 3 else ''
        results = query_by_topic(workspace, topic)
        print(json.dumps(results, indent=2))
    
    elif command == 'wake':
        topic = None
        top_n = 5
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == '--topic' and i + 1 < len(sys.argv):
                topic = sys.argv[i + 1]
            elif arg == '--top' and i + 1 < len(sys.argv):
                top_n = int(sys.argv[i + 1])
        
        results = wake_query(workspace, topic, top_n)
        
        # Print formatted report
        print(format_wake_report(results))
        print("\n--- Raw JSON ---")
        print(json.dumps(results, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()
