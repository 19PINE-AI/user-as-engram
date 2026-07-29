param(
    [string]$ArchivePath = (Join-Path $PSScriptRoot 'AnonymousCodeDataAppendix.zip')
)

$ErrorActionPreference = 'Stop'
$archiveFull = [System.IO.Path]::GetFullPath($ArchivePath)
if (-not (Test-Path -LiteralPath $archiveFull)) {
    throw "Archive does not exist: $archiveFull"
}

$tempParent = [System.IO.Path]::GetTempPath()
$tempRoot = Join-Path $tempParent ('validate-anonymous-appendix-' + [guid]::NewGuid().ToString('N'))
$fixedDate = [datetimeoffset]'2026-01-01T00:00:00Z'

try {
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    Expand-Archive -LiteralPath $archiveFull -DestinationPath $tempRoot
    $root = Join-Path $tempRoot 'anonymous_code_data_appendix'

    $required = @(
        'LICENSE',
        'appendix/README.md',
        'appendix/ANONYMOUS_NOTICE',
        'code/scripts/layered_architecture.py',
        'code/scripts/joint_opt.py',
        'code/nanochat_harness/nanochat/engram_module.py',
        'code/nanochat_harness/LICENSE',
        'data/corpora.json',
        'data/users/u000.json',
        'results/layered_d20_r16_full.json',
        'results/memory_systems_paraphrase.json',
        'results/multihop_d20_n63.json',
        'analysis/make_all_figures.sh'
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $relative))) {
            throw "Required archive file is missing: $relative"
        }
    }

    $paperOnlyDocs = @(
        'appendix/CONFIGURATIONS.md',
        'appendix/DEVELOPMENT_SEARCH.md',
        'appendix/RUNS_AND_RANDOMNESS.md',
        'appendix/METRICS.md',
        'appendix/KNOWN_LIMITATIONS.md'
    )
    foreach ($relative in $paperOnlyDocs) {
        if (Test-Path -LiteralPath (Join-Path $root $relative)) {
            throw "Paper-only checklist material leaked into the appendix: $relative"
        }
    }

    $blockedNames = @(
        '.git', '.github', '.gitignore', '.gitattributes', '.gitmodules',
        '.mailmap', '.git-blame-ignore-revs'
    )
    $blocked = @(Get-ChildItem -LiteralPath $root -Recurse -Force | Where-Object {
        $blockedNames -contains $_.Name
    })
    if ($blocked.Count -gt 0) {
        throw "Git/GitHub metadata remains after extraction: $($blocked[0].FullName)"
    }

    $paperArtifacts = @(Get-ChildItem -LiteralPath $root -Recurse -Force | Where-Object {
        ($_.PSIsContainer -and $_.Name -in 'paper', 'paper_aaai') -or
        (-not $_.PSIsContainer -and ($_.Extension -in '.tex', '.bib', '.bbl', '.pdf'))
    })
    if ($paperArtifacts.Count -gt 0) {
        throw "Paper material remains after extraction: $($paperArtifacts[0].FullName)"
    }

    $patterns = @(
        'C:\\Users\\[^\\\r\n]+',
        'C:/Users/[^/\r\n]+',
        '/home/[^/\r\n]+/user-as-engram',
        '19PINE', 'Pine AI', 'BSQL',
        '01\.me/research/user-as-engram',
        'arXiv:2606\.19172',
        'github\.com/(19PINE-AI|[^/\s]+/user-as-engram)'
    )
    $textExtensions = @(
        '.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.toml', '.lock',
        '.sh', '.yaml', '.yml', '.html', '.svg', '.python-version', ''
    )
    foreach ($file in Get-ChildItem -LiteralPath $root -Recurse -File) {
        if ($textExtensions -notcontains $file.Extension) { continue }
        $raw = Get-Content -Raw -LiteralPath $file.FullName
        foreach ($pattern in $patterns) {
            if ($raw -match $pattern) {
                throw "Identifying content matched '$pattern' after extraction: $($file.FullName)"
            }
        }
    }

    $jsonCount = 0
    foreach ($file in Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.json') {
        try {
            Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json | Out-Null
            $jsonCount++
        }
        catch {
            throw "Invalid JSON after packaging: $($file.FullName): $($_.Exception.Message)"
        }
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($archiveFull)
    try {
        $badTime = @($zip.Entries | Where-Object {
            $_.LastWriteTime.UtcDateTime.Date -ne $fixedDate.UtcDateTime.Date
        })
        if ($badTime.Count -gt 0) {
            throw "ZIP timestamp was not normalized: $($badTime[0].FullName)"
        }
        Write-Output "Validation passed"
        Write-Output "ZIP entries: $($zip.Entries.Count)"
        Write-Output "Parsed JSON files: $jsonCount"
        Write-Output "Archive bytes: $((Get-Item -LiteralPath $archiveFull).Length)"
    }
    finally {
        $zip.Dispose()
    }
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        $resolvedTemp = [System.IO.Path]::GetFullPath($tempRoot)
        $resolvedSystemTemp = [System.IO.Path]::GetFullPath($tempParent)
        if (-not $resolvedTemp.StartsWith($resolvedSystemTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing to remove a validation path outside the system temp directory.'
        }
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
