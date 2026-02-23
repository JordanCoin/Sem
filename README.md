# Sem

Layered `.sem` protocol implementation with:

1. **File layer**: OBJ + `#@ {json}` semantic overlay.
2. **Coherence layer**: triangle strain and confidence dampening.
3. **API layer**: local HTTP runtime (`/v1/...`) for agents and UI.

## Quick Start (Python tools)

```bash
# Convert memory-v2 index to .sem
python3 converters/memory_v2_to_sem.py memory-index.jsonl workspace.sem

# Query for high-strain beliefs (tensions)
python3 tools/sem_query.py workspace.sem strain --top 10

# Wake-up context (for agents recovering from session breaks)
python3 tools/sem_query.py workspace.sem wake --topic "last topic"
```

## Files

- `generated/workspace.sem`: canonical sample workspace file.
- `sem_protocol.c`: low-level generator/validator for the file protocol.
- `sem_runtime.c`: low-level loopback HTTP runtime implementing the API contract.
- `converters/`: Import tools (memory-v2 → .sem)
- `tools/`: Query and analysis tools (strain, neighborhood, wake-up)

## Build

```bash
gcc -std=c11 -O2 -Wall -Wextra -pedantic sem_protocol.c -o sem_protocol
gcc -std=c11 -O2 -Wall -Wextra -pedantic sem_runtime.c -lm -o sem_runtime
```

## Protocol check (file layer)

```bash
./sem_protocol
```

This writes and re-opens `generated/workspace.sem`, then validates required protocol elements.

## Run API runtime (API + coherence layers)

```bash
./sem_runtime
```

Runtime listens on `127.0.0.1:7318` and supports these endpoints from the notes:

- `POST /v1/files/open`
- `POST /v1/files/save`
- `POST /v1/files/validate`
- `GET /v1/beliefs`
- `PATCH /v1/beliefs/{id}`
- `POST /v1/beliefs/{id}/move`
- `GET /v1/triangles`
- `POST /v1/triangles`
- `POST /v1/relax`
- `POST /v1/query`
- `POST /v1/requests/run`
- `POST /v1/patch/apply`

## Quick API smoke test

```bash
curl -sS -X POST http://127.0.0.1:7318/v1/files/open -H 'content-type: application/json' -d '{"path":"generated/workspace.sem"}'
curl -sS 'http://127.0.0.1:7318/v1/beliefs?session_id=s_1'
curl -sS http://127.0.0.1:7318/v1/triangles?session_id=s_1
```
