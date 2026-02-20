# Sem

`.sem` file protocol reference implementation using low-level POSIX file operations.

## What this does

- Reads the specification from `Notes.txt` and materializes a valid `.sem` file.
- Writes `generated/workspace.sem` using `open(2)` + `write(2)` loops.
- Re-opens and validates the generated file using `open(2)` + `read(2)` parsing.
- Confirms protocol essentials: header record, OBJ vertices/faces, semantic `#@` records.

## Build

```bash
gcc -std=c11 -O2 -Wall -Wextra -pedantic sem_protocol.c -o sem_protocol
```

## Run

```bash
./sem_protocol
```

Successful output indicates the generated `.sem` file was opened and validated.
