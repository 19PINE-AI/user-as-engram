param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'AnonymousCodeDataAppendix.zip')
)

$ErrorActionPreference = 'Stop'

$paperDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repoRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $paperDir))
$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
if ((Split-Path -Parent $outputFull) -ne $paperDir) {
    throw 'OutputPath must remain inside paper_aaai.'
}

$tempParent = [System.IO.Path]::GetTempPath()
$tempRoot = Join-Path $tempParent ('anonymous-code-data-appendix-' + [guid]::NewGuid().ToString('N'))
$archiveRoot = Join-Path $tempRoot 'anonymous_code_data_appendix'
$fixedTime = [datetime]'2026-01-01T00:00:00Z'

function Copy-Tree([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required source is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
}

function Copy-AnalysisFiles([string]$Destination) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $paperDir -File | Where-Object {
        $_.Extension -in '.py', '.sh', '.txt' -and $_.Name -ne 'abstract.txt'
    } | Copy-Item -Destination $Destination
}

function Remove-GitMetadata([string]$Root) {
    $blockedNames = @(
        '.git', '.github', '.gitignore', '.gitattributes', '.gitmodules',
        '.mailmap', '.git-blame-ignore-revs'
    )
    Get-ChildItem -LiteralPath $Root -Recurse -Force | Where-Object {
        $blockedNames -contains $_.Name
    } | Sort-Object FullName -Descending | Remove-Item -Recurse -Force
}

function Remove-GeneratedBinaryOutputs([string]$Root) {
    Get-ChildItem -LiteralPath (Join-Path $Root 'results') -Recurse -File |
        Where-Object { $_.Extension -in '.png', '.pdf' } |
        Remove-Item -Force
}

function Remove-EmptyDirectories([string]$Root) {
    Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force |
        Sort-Object { $_.FullName.Length } -Descending |
        Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force).Count -eq 0 } |
        Remove-Item -Force
}

function Resolve-ResultPointers([string]$Root) {
    $resultsDir = Join-Path $Root 'results'
    Get-ChildItem -LiteralPath $resultsDir -Recurse -File | ForEach-Object {
        if ($_.Length -gt 1024) { return }
        $raw = Get-Content -Raw -LiteralPath $_.FullName
        if ($raw -match '^/home/ubuntu/user-as-engram/results/([^\r\n]+)\s*$') {
            $target = Join-Path $resultsDir $Matches[1]
            if (-not (Test-Path -LiteralPath $target)) {
                throw "Result pointer target is missing: $target"
            }
            Copy-Item -LiteralPath $target -Destination $_.FullName -Force
        }
    }
}

function Scrub-LocalPaths([string]$Root) {
    $textExtensions = @(
        '.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.toml', '.lock',
        '.sh', '.yaml', '.yml', '.html', '.svg', '.ps1', '.python-version', ''
    )
    Get-ChildItem -LiteralPath $Root -Recurse -File | ForEach-Object {
        if ($textExtensions -notcontains $_.Extension) { return }
        $raw = Get-Content -Raw -LiteralPath $_.FullName
        $clean = $raw
        $clean = $clean -replace 'C:\\Users\\[^\\\r\n]+\\[^\r\n]*?\\user-as-engram', '$APPENDIX_ROOT'
        $clean = $clean -replace 'C:/Users/[^/\r\n]+/[^\r\n]*?/user-as-engram', '$APPENDIX_ROOT'
        $clean = $clean -replace '/home/[^/\r\n]+/user-as-engram', '$APPENDIX_ROOT'
        if ($clean -ne $raw) {
            [System.IO.File]::WriteAllText(
                $_.FullName, $clean, [System.Text.UTF8Encoding]::new($false)
            )
        }
    }
}

function Anonymize-AppendixLicense([string]$Root) {
    $licensePath = Join-Path $Root 'LICENSE'
    $raw = Get-Content -Raw -LiteralPath $licensePath
    $clean = $raw.Replace('Copyright 2026 Pine AI', 'Copyright 2026 Anonymous Authors')
    if ($clean -eq $raw) {
        throw 'Expected author copyright line was not found in the appendix license copy.'
    }
    [System.IO.File]::WriteAllText(
        $licensePath, $clean, [System.Text.UTF8Encoding]::new($false)
    )
}

function Assert-Anonymous([string]$Root) {
    $blockedPathNames = @('.git', '.github', '.gitignore', '.gitattributes', '.gitmodules')
    $badPaths = @(Get-ChildItem -LiteralPath $Root -Recurse -Force | Where-Object {
        $blockedPathNames -contains $_.Name
    })
    if ($badPaths.Count -gt 0) {
        throw "Git/GitHub metadata remains in the staged appendix: $($badPaths[0].FullName)"
    }

    $paperArtifacts = @(Get-ChildItem -LiteralPath $Root -Recurse -Force | Where-Object {
        ($_.PSIsContainer -and $_.Name -in 'paper', 'paper_aaai') -or
        (-not $_.PSIsContainer -and ($_.Extension -in '.tex', '.bib', '.bbl', '.pdf'))
    })
    if ($paperArtifacts.Count -gt 0) {
        throw "Paper material entered the staged appendix: $($paperArtifacts[0].FullName)"
    }

    $patterns = @(
        'C:\\Users\\[^\\\r\n]+',
        'C:/Users/[^/\r\n]+',
        '/home/[^/\r\n]+/user-as-engram',
        '19PINE',
        'Pine AI',
        'BSQL',
        '01\.me/research/user-as-engram',
        'arXiv:2606\.19172',
        'github\.com/(19PINE-AI|[^/\s]+/user-as-engram)'
    )
    $textExtensions = @(
        '.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.toml', '.lock',
        '.sh', '.yaml', '.yml', '.html', '.svg', '.ps1', '.python-version', ''
    )
    foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File) {
        if ($textExtensions -notcontains $file.Extension) { continue }
        $raw = Get-Content -Raw -LiteralPath $file.FullName
        foreach ($pattern in $patterns) {
            if ($raw -match $pattern) {
                throw "Identifying content matched '$pattern' in $($file.FullName)"
            }
        }
    }
}

function Normalize-Metadata([string]$Root) {
    # Set children before parents because touching a child can update its
    # directory's timestamp on Windows.
    Get-ChildItem -LiteralPath $Root -Recurse -Force |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
        $_.CreationTimeUtc = $fixedTime
        $_.LastWriteTimeUtc = $fixedTime
        $_.LastAccessTimeUtc = $fixedTime
        }
    (Get-Item -LiteralPath $Root).CreationTimeUtc = $fixedTime
    (Get-Item -LiteralPath $Root).LastWriteTimeUtc = $fixedTime
    (Get-Item -LiteralPath $Root).LastAccessTimeUtc = $fixedTime
}

try {
    New-Item -ItemType Directory -Path $archiveRoot | Out-Null

    Copy-Tree (Join-Path $repoRoot 'code') (Join-Path $archiveRoot 'code')
    Copy-Tree (Join-Path $repoRoot 'data') (Join-Path $archiveRoot 'data')
    Copy-Tree (Join-Path $repoRoot 'results') (Join-Path $archiveRoot 'results')
    Copy-Tree (Join-Path $repoRoot 'appendix') (Join-Path $archiveRoot 'appendix')
    Copy-AnalysisFiles (Join-Path $archiveRoot 'analysis')
    Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination (Join-Path $archiveRoot 'LICENSE')

    # Checklist items outside 4.3--4.6 are answered in the paper, not through
    # the submitted code/data appendix. Keep author working notes in the
    # repository but omit them from the staged archive.
    @(
        'CONFIGURATIONS.md',
        'DEVELOPMENT_SEARCH.md',
        'RUNS_AND_RANDOMNESS.md',
        'METRICS.md',
        'KNOWN_LIMITATIONS.md'
    ) | ForEach-Object {
        $path = Join-Path (Join-Path $archiveRoot 'appendix') $_
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }

    Remove-GitMetadata $archiveRoot
    Remove-GeneratedBinaryOutputs $archiveRoot
    Remove-EmptyDirectories $archiveRoot
    Resolve-ResultPointers $archiveRoot
    Scrub-LocalPaths $archiveRoot
    Anonymize-AppendixLicense $archiveRoot
    Assert-Anonymous $archiveRoot
    Normalize-Metadata $archiveRoot

    if (Test-Path -LiteralPath $outputFull) {
        Remove-Item -LiteralPath $outputFull -Force
    }
    Compress-Archive -LiteralPath $archiveRoot -DestinationPath $outputFull -CompressionLevel Optimal

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($outputFull)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -match '(^|/)(\.git|\.github)(/|$)' -or
                $entry.FullName -match '(^|/)\.git(ignore|attributes|modules)$') {
                throw "Git/GitHub metadata entered ZIP: $($entry.FullName)"
            }
            if ($entry.FullName -match '(^|/)(paper|paper_aaai)(/|$)' -or
                $entry.FullName -match '\.(tex|bib|bbl|pdf)$') {
                throw "Paper material entered ZIP: $($entry.FullName)"
            }
        }
        Write-Output "Created $outputFull"
        Write-Output "Entries: $($zip.Entries.Count)"
        Write-Output "Bytes: $((Get-Item -LiteralPath $outputFull).Length)"
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
            throw 'Refusing to remove a temporary path outside the system temp directory.'
        }
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
