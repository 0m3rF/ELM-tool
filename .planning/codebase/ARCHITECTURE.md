# Architecture

## Core Patterns
The ELM (Extract, Load, Mask) tool follows a standard modular pipeline architecture separated by distinct command flows:
- **Presentation Layer**: A CLI built with `click`.
- **Coordinator Layer**: Command modules located in `elm/elm_commands/` orchestrate logic across domains.
- **Core Engine Layer**: Defines the core operations (copying, masking, generating) located in `elm/core/`.
- **Data Access Utilities**: Abstracts database operations and interactions in `elm/elm_utils/`.

## Data Flow
1. **User Input**: Received via `elm.py` / `cli.py` and routed to specific command handlers in `elm_commands`.
2. **Environment Configuration**: Processed by `environment.py` and `config.py` in `core` establishing connections via `db_utils`.
3. **Execution Pipeline**:
   - Extraction: Reads metadata and schema using `metadata_sync.py` and extracts tables.
   - Masking: Applies algorithms from `masking.py` backed by `Faker` logic if masking is enabled.
   - Loading: Persists back to the target database directly or streams it.

## Key Abstractions
- **Environments**: Configurations tying together database connections.
- **Masking Algorithms**: Abstracted text modification functions.
