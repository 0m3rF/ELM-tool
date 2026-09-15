# Canonical acceptance fixtures

The checked-in CSV and NDJSON rows are human-reviewable seeds. Integration fixture loaders expand
them into narrow rows, 64-column wide rows, nulls, Unicode, Decimal128(38,9), nanosecond UTC and
offset timestamps, binary values from zero bytes through 1 MiB, and string/binary LOBs through 64
MiB. Parquet fixtures are generated from Arrow at test time so their embedded schema cannot drift
from the current canonical schema.

No fixture contains production data or credentials. The 10 GB and 100 GB datasets are deterministic
expansions identified by generator version and BLAKE3 digest; generated files are not committed.

