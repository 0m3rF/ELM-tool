$ErrorActionPreference = 'Stop'
$previousMatrixPassword = $env:ELM_MATRIX_PASSWORD
# Oracle XE rejects passwords longer than 30 bytes; retain SQL Server complexity.
$env:ELM_MATRIX_PASSWORD = 'Elm-' + [Guid]::NewGuid().ToString('N').Substring(0, 24)
$fixtureArgs = @('compose', '-p', 'elm-cross-native', '-f', 'tests/cross-database/compose.yml', '-f', 'tests/cross-database/native.yml')
try {
    & docker @fixtureArgs build tests
    if ($LASTEXITCODE -ne 0) { throw 'Native matrix image build failed' }
    # Compile before starting four databases to reduce peak memory during cold builds.
    & docker @fixtureArgs run --rm --no-deps tests cargo test --locked -p elm-connectors --features all-databases --test cross_database --no-run
    if ($LASTEXITCODE -ne 0) { throw 'Native matrix compilation failed' }
    & docker @fixtureArgs up -d --wait --wait-timeout 600 postgres mysql oracle sqlserver
    if ($LASTEXITCODE -ne 0) {
        & docker @fixtureArgs logs --no-color --tail 60 postgres mysql oracle sqlserver |
            ForEach-Object { $_.Replace($env:ELM_MATRIX_PASSWORD, '[REDACTED]') }
        throw 'Native matrix database startup failed'
    }
    & docker @fixtureArgs run --rm tests cargo test --locked -p elm-connectors --features all-databases --test cross_database same_table -- --ignored --test-threads=1
    if ($LASTEXITCODE -ne 0) { throw 'Same-table acceptance failed' }
    & docker @fixtureArgs run --rm tests
    if ($LASTEXITCODE -ne 0) { throw 'Native matrix acceptance failed' }
} finally {
    & docker @fixtureArgs down --volumes
    $env:ELM_MATRIX_PASSWORD = $previousMatrixPassword
    if ($LASTEXITCODE -ne 0) { throw 'Native matrix fixture cleanup failed' }
}
