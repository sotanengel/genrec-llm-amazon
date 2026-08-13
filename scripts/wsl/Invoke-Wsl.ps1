<#
.SYNOPSIS
  Windows-side caller for running commands inside the Ubuntu-24.04 WSL distro's
  genrec-llm-amazon environment.

.DESCRIPTION
  Wraps wsl.exe with the PowerShell 5.1 conventions required on this machine:
    - Sets [Console]::OutputEncoding to Unicode before calling `wsl -l -q`
      (which emits UTF-16LE and renders as mojibake under PS 5.1 otherwise),
      then restores it for readable output from the command itself.
    - Passes --cd with a *Linux* path, so WSL does not inherit the Windows
      current directory (which would start the shell in /mnt/c/... and break
      every relative path in the repo).
    - Checks $LASTEXITCODE explicitly after every native call, since
      PowerShell 5.1 has no short-circuit chaining operators and a
      ';'-chain only propagates the exit code of the last command.

.PARAMETER Command
  The command line to run inside WSL via `bash -lc`, e.g.
  "scripts/wsl/run.sh pytest -x tests/".

.PARAMETER Distro
  WSL distro name. Default: Ubuntu-24.04.

.PARAMETER User
  Linux username inside the distro. Default: lowercased $env:USERNAME --
  override with -User if install_distro.ps1 was run with a different -LinuxUser.

.PARAMETER RepoPath
  Linux path (inside WSL) to the repo clone. Default: /home/<User>/src/genrec-llm-amazon.

.EXAMPLE
  .\Invoke-Wsl.ps1 -Command "scripts/wsl/doctor.sh"

.EXAMPLE
  .\Invoke-Wsl.ps1 -Command "scripts/wsl/run.sh python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command,

    [string]$Distro = "Ubuntu-24.04",

    [string]$User = $env:USERNAME.ToLowerInvariant(),

    [string]$RepoPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $RepoPath = "/home/$User/src/genrec-llm-amazon"
}

$previousEncoding = [Console]::OutputEncoding

try {
    Write-Host "Checking distro '$Distro' is registered..."
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $registered = & wsl.exe -l -q
    $registeredExit = $LASTEXITCODE
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    if ($registeredExit -ne 0) {
        Write-Error "wsl.exe -l -q failed with exit code $registeredExit. Is WSL installed?"
        exit $registeredExit
    }

    $names = @($registered | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($names -notcontains $Distro) {
        Write-Error "Distro '$Distro' is not registered. Run scripts\wsl\install_distro.ps1 first."
        exit 1
    }

    Write-Host "Running in WSL (distro=$Distro, user=$User, cwd=$RepoPath):"
    Write-Host "  $Command"

    & wsl.exe -d $Distro -u $User --cd $RepoPath -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed inside WSL with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}
finally {
    [Console]::OutputEncoding = $previousEncoding
}
