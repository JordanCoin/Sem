# Principled Strain Objective & Validation Experiment Design

**Document Type:** Research Design  
**Status:** Draft for Review  
**Date:** February 2026

---

## Part 1: Toward a Principled Strain Objective

### 1.1 The Current Problem

Our strain computation is:

```
Total = Semantic + Temporal + Relational
      = w_s × S + w_t × T + w_r × R
```

The reviewer correctly identified: **this is a weighted additive heuristic, not a physical constraint system.**

The weights (0.3 for temporal, 0.8 for contradicts, etc.) are arbitrary. We have no principled basis for them.

### 1.2 Constraint Satisfaction Framework

Paul Thagard's work on "Coherence as Constraint Satisfaction" provides a rigorous foundation:

**Core idea:** Coherence is about dividing elements into "accepted" and "rejected" sets such that constraints are maximally satisfied.

- **Positive constraint:** Two elements that cohere (should both be accepted or both rejected)
- **Negative constraint:** Two elements that incohere (one should be accepted, one rejected)

**Energy function:**

```
E = Σ w_ij × (a_i ⊕ a_j)  for negative constraints
  + Σ w_ij × (a_i ⊙ a_j)  for positive constraints
```

Where:
- `a_i ∈ {0, 1}` = acceptance state
- `w_ij` = constraint strength
- Minimizing E = maximizing coherence

### 1.3 Mapping to .sem

Our edges ARE constraints:
- **"elaborates"** = positive constraint (should agree)
- **"contradicts"** = negative constraint (should disagree)
- **"related"** = weak positive constraint

**Proposed energy function:**

```python
def compute_energy(beliefs, edges):
    E = 0
    for edge in edges:
        a_i = beliefs[edge.source].accepted  # 0 or 1
        a_j = beliefs[edge.target].accepted  # 0 or 1
        
        if edge.relation == "contradicts":
            # Negative constraint: satisfied if exactly one accepted
            satisfied = (a_i != a_j)
            E += edge.weight * (1 - satisfied)
        else:
            # Positive constraint: satisfied if both same state
            satisfied = (a_i == a_j)
            E += edge.weight * (1 - satisfied)
    
    return E  # Lower = more coherent
```

**Strain as local energy contribution:**

```python
def belief_strain(belief, edges):
    """
    Strain = how much this belief contributes to total energy
    when currently in "accepted" state.
    """
    contribution = 0
    for edge in incident_edges(belief):
        other = get_other_belief(edge, belief)
        
        if edge.relation == "contradicts":
            # Violated if both accepted
            if belief.accepted and other.accepted:
                contribution += edge.weight
        else:
            # Violated if states differ
            if belief.accepted != other.accepted:
                contribution += edge.weight
    
    return contribution
```

This is **principled**: strain measures how much removing/changing this belief would reduce total incoherence.

### 1.4 Weight Derivation

Instead of arbitrary weights, derive from data:

**Option A: Cosine distance as weight**
```python
edge.weight = cosine_distance(emb_a, emb_b)
# Closer beliefs have stronger constraints
```

**Option B: Learned weights**
```python
# Train on user feedback: which high-strain items were actually conflicts?
# Optimize weights to predict human-labeled conflicts
```

**Option C: Relation-type priors from literature**
```python
# Base on Thagard's empirical studies of explanatory coherence
relation_weights = {
    "contradicts": 1.0,    # Strong negative
    "supersedes": 0.7,     # Moderate negative (old should be rejected)
    "elaborates": 0.3,     # Weak positive
    "related": 0.2,        # Weak positive
}
```

### 1.5 Aggregation: Max vs Mean vs Energy

The reviewer noted: **mean dilutes strong signals.**

**Proposed: Use max with decay**

```python
def belief_strain_v2(belief, edges):
    strains = [edge_strain(edge, belief) for edge in incident_edges(belief)]
    
    if not strains:
        return 0.15  # Isolation penalty
    
    # Max captures the worst violation
    max_strain = max(strains)
    
    # Mean captures distributed tension
    mean_strain = sum(strains) / len(strains)
    
    # Combine: max-weighted average
    # α controls how much we favor max vs mean
    α = 0.7
    return α * max_strain + (1 - α) * mean_strain
```

This ensures a single strong contradiction isn't hidden by many weak elaborations.

---

## Part 2: Validation Experiment Design

### 2.1 Research Question

**Does strain-based wake orientation improve agent performance compared to flat memory retrieval?**

Specifically:
1. Does high-strain surfacing reduce time-to-resolution for open decisions?
2. Does it reduce contradictory outputs across sessions?
3. Does it improve planning consistency?

### 2.2 Experimental Setup

#### Conditions

| Condition | Memory System | Wake Context |
|-----------|--------------|--------------|
| **Control** | Flat memory-v2 | Top-k by recency + importance |
| **Baseline Heuristic** | Flat memory-v2 | Top-k by (age × importance) |
| **Treatment** | .sem mesh | Top-k by strain score |

#### Task Types

**Type A: Decision Resolution**
- Plant N open decisions in memory (e.g., "Should prioritize video or blog?")
- After M session gaps, ask agent to resolve
- Measure: Turns to resolution, consistency with prior context

**Type B: Contradiction Detection**
- Plant contradictory beliefs (A: "Video is primary", B: "Blog is primary")
- Measure: Does agent notice contradiction? How many turns?

**Type C: Planning Coherence**
- Give agent a multi-step project plan
- After session gap, ask for status update
- Measure: Consistency with prior plan, hallucination rate

### 2.3 Metrics

| Metric | Definition | How to Measure |
|--------|------------|----------------|
| **Resolution Latency** | Turns until decision is made | Count turns |
| **Contradiction Awareness** | Did agent surface conflicting beliefs? | Manual label (0/1) |
| **Planning Consistency** | Does new output align with prior plan? | LLM-as-judge (1-5 scale) |
| **Hallucination Rate** | Facts not grounded in memory | Manual count |
| **User Preference** | Which response is more helpful? | A/B blind evaluation |

### 2.4 Dataset Construction

**Synthetic Memory Index:**

Create a controlled memory index with:
- 50 decisions (varying importance, some open, some resolved)
- 20 explicit contradictions (pairs with "contradicts" edges)
- 100 events (temporal spread: 1 day to 30 days ago)
- 30 learnings (grounded facts)

**Edge Structure:**
- Auto-inferred "related" edges (cosine > 0.55)
- Manual "contradicts" edges for planted conflicts
- Manual "supersedes" edges for resolved decisions

### 2.5 Protocol

```
For each task in [A, B, C]:
    For each condition in [Control, Baseline, Treatment]:
        For each trial in [1..N]:  # N = 20 per condition
            
            1. Initialize fresh session
            2. Inject wake context (per condition)
            3. Present task prompt
            4. Record agent response
            5. Continue for up to 5 turns
            6. Measure metrics
            
    Compute aggregate statistics
    Run significance tests (t-test or Mann-Whitney U)
```

### 2.6 Hypotheses

**H1:** Treatment (strain-based) will have lower Resolution Latency than Control  
**H2:** Treatment will have higher Contradiction Awareness than Control  
**H3:** Treatment will have higher Planning Consistency than Baseline Heuristic  
**H4:** Treatment will have lower Hallucination Rate than Control  

### 2.7 Expected Outcomes

| If True | Implication |
|---------|-------------|
| H1 passes | Strain surfaces relevant conflicts faster |
| H2 passes | Geometric structure captures contradictions |
| H3 passes | Strain beats naive age×importance heuristic |
| H4 passes | Structured context reduces hallucination |

**If all pass:** .sem provides measurable value over flat memory.

**If H3 fails:** We're building elegant machinery on top of something simpler.

### 2.8 Practical Implementation

**Phase 1: Synthetic Validation (1 week)**
- Create synthetic memory index
- Implement all three conditions
- Run automated trials with GPT-4 as agent
- Compute metrics

**Phase 2: Real-World Pilot (2 weeks)**
- Run on actual Stargazer memory index
- Track resolution of real open decisions
- A/B test wake reports with Doe as evaluator

**Phase 3: Publication (if warranted)**
- Write up results
- Compare to LoCoMo, MemBench, CloneMem benchmarks
- Submit to agent memory workshop/venue

---

## Part 3: Open Questions for Reviewer

Given their offer to help formalize:

1. **Energy function choice:** Is the constraint satisfaction framing correct, or should we use a continuous energy model (like Hopfield networks)?

2. **Weight learning:** If we learn weights from human feedback, what's the right loss function? Cross-entropy on "is this actually a conflict"?

3. **Aggregation function:** Is max-weighted mean principled, or is there a better choice (e.g., softmax attention over strains)?

4. **Temporal integration:** Should temporal strain be part of the energy function, or a separate signal?

5. **Acceptance states:** We currently treat all beliefs as "accepted." Should we model explicit accept/reject states and minimize energy over that partition?

---

## Summary

**Principled Strain:**
- Reframe as constraint satisfaction energy minimization
- Strain = local energy contribution = "how much does this belief cause incoherence"
- Derive weights from data or principled priors
- Use max-weighted aggregation to preserve strong signals

**Validation:**
- Compare strain-based wake vs flat retrieval vs naive heuristic
- Measure resolution latency, contradiction awareness, planning consistency
- Synthetic first, then real-world pilot

**Next Steps:**
1. Implement energy-based strain computation
2. Build synthetic validation dataset
3. Run experiments
4. Iterate on weights/aggregation based on results

---

*"The next milestone is not more geometry. It is: run experiments."*  
— Anonymous Reviewer
