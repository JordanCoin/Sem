#!/usr/bin/env python3
"""Quick validation: do 3D positions preserve semantic structure?"""
import json
import math
import sys

def load_beliefs(sem_path):
    beliefs = []
    vertices = []
    
    with open(sem_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('#@ '):
                try:
                    rec = json.loads(line[3:])
                    if rec.get('type') == 'belief':
                        v_idx = rec.get('vertex', 0) - 1
                        if 0 <= v_idx < len(vertices):
                            rec['position'] = vertices[v_idx]
                        beliefs.append(rec)
                except:
                    pass
    return beliefs

def distance(p1, p2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

def find_nearest(belief, all_beliefs, n=5):
    """Find n nearest beliefs by 3D position."""
    if 'position' not in belief:
        return []
    
    distances = []
    for b in all_beliefs:
        if b['id'] != belief['id'] and 'position' in b:
            d = distance(belief['position'], b['position'])
            distances.append((d, b))
    
    distances.sort(key=lambda x: x[0])
    return distances[:n]

if __name__ == '__main__':
    sem_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/jordanjackson/.openclaw/workspace/Sem/generated/stargazer-memory.sem'
    
    beliefs = load_beliefs(sem_path)
    print(f"Loaded {len(beliefs)} beliefs with positions")
    print()
    
    # Test: find a recent belief and check its neighbors
    recent = [b for b in beliefs if '2026-02-22' in b.get('updated_at', '') or '2026-02-23' in b.get('updated_at', '')]
    
    if recent:
        test = recent[0]
        print(f"TEST BELIEF: {test['proposition'][:80]}...")
        print(f"  tags: {test.get('tags', [])}")
        print()
        
        nearest = find_nearest(test, beliefs)
        print("NEAREST 5 IN 3D SPACE:")
        for d, b in nearest:
            print(f"  dist={d:.2f} | {b['proposition'][:60]}...")
            print(f"           tags: {b.get('tags', [])[:3]}")
        print()
        
        # Also test a random older one
        if len(beliefs) > 100:
            test2 = beliefs[50]
            print(f"TEST BELIEF 2: {test2['proposition'][:80]}...")
            print(f"  tags: {test2.get('tags', [])}")
            print()
            
            nearest2 = find_nearest(test2, beliefs)
            print("NEAREST 5:")
            for d, b in nearest2:
                print(f"  dist={d:.2f} | {b['proposition'][:60]}...")
