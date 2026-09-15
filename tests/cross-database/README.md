# Cross-database acceptance fixtures

The first slice covers PostgreSQL 16 ↔ MySQL 8.4 using the real transfer engine. It verifies
Unicode, binary bytes, exact decimal scale/value, microsecond timestamps, NULLs, staged APPEND,
new-target REPLACE, FAIL preserving an existing target, and empty-source REPLACE preserving schema.
These two directions are not evidence for the other 14 database pairs or full type coverage.

Fixtures bind only loopback ports 55441 and 53341. Use a generated disposable `ELM_MATRIX_PASSWORD`.
Never substitute production databases. The ignored tests create uniquely named tables in `elm_matrix`;
fixture teardown removes their journals and data. Tests do not read saved ELM connections or keychains.

From the repository root:

```powershell
$env:ELM_MATRIX_PASSWORD = 'Elm-' + [Guid]::NewGuid().ToString('N')
docker compose -p elm-cross-acceptance -f tests/cross-database/compose.yml up -d --wait --wait-timeout 180
cargo test -p elm-connectors --features postgresql,mysql --test cross_database -- --ignored --test-threads=1
docker compose -p elm-cross-acceptance -f tests/cross-database/compose.yml down --volumes
```

Always tear down this named fixture after testing, including after a failure. Do not use broad
Docker cleanup/prune commands; other projects may share the Docker daemon. Keep the generated
password out of reports. Linux CI runs the same fixture and test command.
