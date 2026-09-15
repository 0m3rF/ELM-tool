# Requires the dedicated fixture described in benchmarks/README.md.
param([switch]$InstalledElm, [string]$PythonExecutable = 'C:/Users/omerf/AppData/Local/Python/pythoncore-3.14-64/python.exe')
$ErrorActionPreference = 'Stop'
$benchmarkRoot = Split-Path -Parent $PSScriptRoot
$benchmarkExecutable = Join-Path $benchmarkRoot 'target/release/examples/postgres_performance.exe'
if ($InstalledElm) {
    $benchmarkExecutable = $PythonExecutable
}
if (-not (Test-Path -LiteralPath $benchmarkExecutable)) {
    throw 'Build the release postgres_performance example first.'
}
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $benchmarkExecutable
$startInfo.WorkingDirectory = $benchmarkRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.Environment['ELM_PERF_DISPOSABLE'] = 'yes'
if ($InstalledElm) {
    $startInfo.ArgumentList.Add('-I')
    $startInfo.ArgumentList.Add('-X')
    $startInfo.ArgumentList.Add('utf8')
    $startInfo.ArgumentList.Add((Join-Path $PSScriptRoot 'installed_elm_performance.py'))
}
$benchmarkProcess = [System.Diagnostics.Process]::new()
$benchmarkProcess.StartInfo = $startInfo
$benchmarkPeakBytes = 0L
$benchmarkDeadline = [DateTime]::UtcNow.AddMinutes(15)
try {
    if (-not $benchmarkProcess.Start()) { throw 'Unable to start benchmark.' }
    $benchmarkOutput = $benchmarkProcess.StandardOutput.ReadToEndAsync()
    $benchmarkErrors = $benchmarkProcess.StandardError.ReadToEndAsync()
    $benchmarkMeasuredProcess = $benchmarkProcess
    if ($InstalledElm) {
        # A Windows venv python.exe may be a tiny launcher for the real interpreter.
        # Measure that worker, not the launcher, while retaining the launcher's exit status.
        $null = $benchmarkProcess.WaitForExit(200)
        $pythonChildren = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($benchmarkProcess.Id)" |
            Where-Object { $_.Name -eq 'python.exe' })
        if ($pythonChildren.Count -eq 1) {
            $benchmarkMeasuredProcess = Get-Process -Id $pythonChildren[0].ProcessId
        } elseif ($pythonChildren.Count -gt 1) {
            throw 'Multiple Python workers found; memory accounting requires review.'
        }
    }
    while (-not $benchmarkProcess.WaitForExit(100)) {
        if (-not $benchmarkMeasuredProcess.HasExited) {
            $benchmarkMeasuredProcess.Refresh()
            $benchmarkPeakBytes = [Math]::Max($benchmarkPeakBytes, $benchmarkMeasuredProcess.PeakWorkingSet64)
        }
        if ([DateTime]::UtcNow -gt $benchmarkDeadline) {
            $benchmarkProcess.Kill()
            throw 'Benchmark exceeded the 15-minute deadline.'
        }
    }
    $benchmarkProcess.WaitForExit()
    $diagnostics = $benchmarkErrors.GetAwaiter().GetResult()
    if ($diagnostics) { [Console]::Error.WriteLine($diagnostics) }
    if ($benchmarkProcess.ExitCode -ne 0) { throw 'Benchmark failed; no performance result accepted.' }
    $benchmarkReport = $benchmarkOutput.GetAwaiter().GetResult() | ConvertFrom-Json
    $benchmarkReport | Add-Member -NotePropertyName peak_working_set_bytes -NotePropertyValue $benchmarkPeakBytes
    $benchmarkReport | Add-Member -NotePropertyName memory_measurement -NotePropertyValue 'Windows actual worker PeakWorkingSet64 high-water mark queried every 100ms; includes setup/verification, excludes database server and any venv launcher; may miss final interval.'
    $benchmarkReport | Add-Member -NotePropertyName host_cpu -NotePropertyValue $env:PROCESSOR_IDENTIFIER
    $benchmarkReport | Add-Member -NotePropertyName host_logical_processors -NotePropertyValue ([Environment]::ProcessorCount)
    $benchmarkReport | Add-Member -NotePropertyName host_os -NotePropertyValue ([Environment]::OSVersion.VersionString)
    $benchmarkReport | Add-Member -NotePropertyName executable_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath $benchmarkExecutable -Algorithm SHA256).Hash)
    $benchmarkReport | ConvertTo-Json -Depth 10
} finally {
    $benchmarkProcess.Dispose()
}
