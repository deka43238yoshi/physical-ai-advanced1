$ErrorActionPreference = 'Stop'
$commit = 'f3b6013629c7b94f8ee93fd977b161fa8350f9bc'
$repository = 'google-deepmind/mujoco_menagerie'
$prefix = 'robotstudio_so101'
$destination = Join-Path $PSScriptRoot '..\assets\robotstudio_so101'
if (Test-Path -LiteralPath $destination) {
    throw "Existing destination will not be overwritten: $destination"
}
$headers = @{ 'User-Agent' = 'so101-dice-sim' }
$api = "https://api.github.com/repos/$repository/contents"
$entries = Invoke-RestMethod -Uri "$api/$prefix`?ref=$commit" -Headers $headers
$assets = Invoke-RestMethod -Uri "$api/$prefix/assets`?ref=$commit" -Headers $headers
$root = Invoke-RestMethod -Uri "$api`?ref=$commit" -Headers $headers
$files = @($entries | Where-Object { $_.name -in @('so101.xml', 'LICENSE', 'README.md', 'CHANGELOG.md') })
$files += @($assets | Where-Object { $_.type -eq 'file' -and $_.name -like '*.stl' })
$files += @($root | Where-Object { $_.name -eq 'NOTICE' -and $_.type -eq 'file' })
New-Item -ItemType Directory -Path (Join-Path $destination 'assets') | Out-Null
$manifest = @()
foreach ($file in $files) {
    $relative = $file.path
    if ($relative.StartsWith("$prefix/")) {
        $relative = $relative.Substring($prefix.Length + 1)
    }
    $target = Join-Path $destination $relative.Replace('/', '\')
    if (Test-Path -LiteralPath $target) {
        throw "Existing file will not be overwritten: $target"
    }
    $url = "https://raw.githubusercontent.com/$repository/$commit/$($file.path)"
    Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -Headers $headers
    if ((Get-Item -LiteralPath $target).Length -ne $file.size) {
        throw "Size mismatch: $target"
    }
    $manifest += [pscustomobject]@{
        path = $relative
        sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        upstream_git_blob_sha = $file.sha
    }
}
[pscustomobject]@{
    repository = "https://github.com/$repository"
    commit = $commit
    upstream_path = $prefix
    license = 'Apache-2.0'
    modified_upstream_files = @()
    files = $manifest
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination 'provenance.json') -Encoding UTF8
Write-Output "Downloaded $($files.Count) files to $destination"
