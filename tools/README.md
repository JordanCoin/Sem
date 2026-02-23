# Sem Tools

Runtime tools for working with .sem files.

## sem_query.py

Query a .sem workspace for strain, neighborhoods, and wake-up context.

### Commands

```bash
# Find high-strain beliefs (tensions/contradictions)
python3 sem_query.py workspace.sem strain --top 10

# Get neighborhood around a belief
python3 sem_query.py workspace.sem neighborhood <belief_id> --radius 2

# Find recently updated beliefs
python3 sem_query.py workspace.sem recent --since 2026-02-20

# Search by topic
python3 sem_query.py workspace.sem topic "cron delivery"

# Wake-up query (combines all of the above)
python3 sem_query.py workspace.sem wake --topic "last topic" --top 5
```

### Wake-Up Query

The `wake` command is designed for "blink recovery" — helping an agent understand what needs attention after a session break.

It returns:
- **stats** — workspace size and average strain
- **high_strain** — beliefs under tension that need reconciling
- **recent** — recently updated beliefs
- **topic_context** — if topic provided, relevant beliefs
- **topic_neighborhood** — graph neighborhood around best topic match

### Understanding Strain

Strain measures how much an edge has stretched from its rest length:

```
strain = (current_length - rest_length) / rest_length
```

High strain indicates:
- Beliefs that have drifted apart semantically
- Contradictions that need reconciling
- Areas of the workspace that are under tension

Effective confidence is dampened by strain:
```
confidence_effective = confidence_base * exp(-α * strain)
```

This means beliefs in tense regions are automatically less trusted until reconciled.

### Requirements

- Python 3.8+
- No external dependencies (uses only stdlib + math)
