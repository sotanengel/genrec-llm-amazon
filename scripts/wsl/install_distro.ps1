<#
.SYNOPSIS
  Idempotently register and non-interactively provision the Ubuntu-24.04 WSL2
  distro used by this repo.

.DESCRIPTION
  Exits 0 early if the distro is already registered (idempotent -- safe to
  re-run). Otherwise:
    1. `wsl --install -d <Distro> --no-launch` (kernel + optional Windows
       features are assumed already present on this machine; this step
       installs the store-backed distro package only).
    2. Provisions a Linux user non-interactively, running as root via
       `wsl.exe -d <Distro> -u root -- bash -lc '...'`, because the normal
       first-launch OOBE prompts for a username/password on stdin and would
       hang a scripted/unattended run.
    3. Writes /etc/wsl.conf: [user] default=<user>, [boot] systemd=true,
       [interop] appendWindowsPath=false.
    4. `wsl --terminate <Distro>` so the new /etc/wsl.conf takes effect.
    5. Re-reads `wsl -l -v` and asserts the distro's WSL VERSION is 2.

  wsl.exe emits UTF-16LE for `-l -v` / `-l -q` / `--status`; under PowerShell
  5.1 that renders as mojibake unless [Console]::OutputEncoding is switched to
  Unicode before the call (restored afterward).

.PARAMETER Distro
  WSL distro name to install/provision. Default: Ubuntu-24.04.

.PARAMETER LinuxUser
  Linux username to create inside the distro. Default: lowercased $env:USERNAME.

.EXAMPLE
  .\install_distro.ps1

.EXAMPLE
  .\install_distro.ps1 -Distro Ubuntu-24.04 -LinuxUser sota
#>
[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$LinuxUser = $env:USERNAME.ToLowerInvariant()
)

$ErrorActionPreference = "Stop"

function Get-RegisteredDistros {
    $previous = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    try {
        $raw = & wsl.exe -l -q
        if ($LASTEXITCODE -ne 0) {
            throw "wsl.exe -l -q failed with exit code $LASTEXITCODE"
        }
        return @($raw | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    }
    finally {
        [Console]::OutputEncoding = $previous
    }
}

Write-Host "Checking for already-registered WSL distros..."
$existing = Get-RegisteredDistros
if ($existing -contains $Distro) {
    Write-Host "'$Distro' is already registered. Nothing to do."
    exit 0
}

Write-Host "'$Distro' not found among: $($existing -join ', '). Installing..."
& wsl.exe --install -d $Distro --no-launch
if ($LASTEXITCODE -ne 0) {
    Write-Error "wsl --install -d $Distro --no-launch failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

# The store-backed distro install should not require admin rights on a machine
# that already has the WSL/VM Platform optional features enabled and a default
# WSL version set (both true here). If it silently required elevation anyway,
# fail loudly with an actionable message instead of continuing into a broken
# provisioning step below.
$afterInstall = Get-RegisteredDistros
if ($afterInstall -notcontains $Distro) {
    Write-Error "'$Distro' is still not registered after 'wsl --install'. This can happen if the install needs Administrator rights or a pending reboot (e.g. WSL/VM Platform features were just enabled). Re-run this script from an elevated PowerShell prompt, or reboot and re-run."
    exit 1
}

Write-Host "Provisioning user '$LinuxUser' non-interactively as root..."

# Runs as root inside the freshly-installed distro. Uses useradd instead of the
# interactive first-launch OOBE (which prompts for a username/password on
# stdin and would hang a scripted run).
$provisionScript = @"
set -euo pipefail
USERNAME='$LinuxUser'
if ! id -u "\$USERNAME" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "\$USERNAME"
fi
mkdir -p /etc/sudoers.d
echo "\$USERNAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-\$USERNAME"
chmod 0440 "/etc/sudoers.d/90-\$USERNAME"
cat > /etc/wsl.conf <<'WSLCONF'
[user]
default=$LinuxUser
[boot]
systemd=true
[interop]
appendWindowsPath=false
WSLCONF
"@

& wsl.exe -d $Distro -u root -- bash -lc $provisionScript
if ($LASTEXITCODE -ne 0) {
    Write-Error "Non-interactive provisioning failed inside '$Distro' with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "Terminating '$Distro' to apply /etc/wsl.conf (systemd, default user)..."
& wsl.exe --terminate $Distro
if ($LASTEXITCODE -ne 0) {
    Write-Error "wsl --terminate $Distro failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "Verifying WSL version for '$Distro'..."
$previous = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::Unicode
try {
    $verboseList = & wsl.exe -l -v
    if ($LASTEXITCODE -ne 0) {
        throw "wsl.exe -l -v failed with exit code $LASTEXITCODE"
    }
}
finally {
    [Console]::OutputEncoding = $previous
}

$distroLine = $verboseList | Where-Object { $_ -match [regex]::Escape($Distro) } | Select-Object -First 1
if (-not $distroLine) {
    Write-Error "Could not find '$Distro' in 'wsl -l -v' output after provisioning: $($verboseList -join ' | ')"
    exit 1
}

$fields = ($distroLine -replace "^\*", "").Trim() -split "\s+"
$version = $fields[-1]
if ($version -ne "2") {
    Write-Error "'$Distro' is not WSL VERSION 2 (got '$version'). Line: $distroLine"
    exit 1
}

Write-Host "'$Distro' is registered, provisioned, and running WSL2 (user: $LinuxUser)."
Write-Host "Next: run scripts\wsl\Invoke-Wsl.ps1 with bootstrap.sh, e.g.:"
Write-Host "  wsl.exe -d $Distro -u $LinuxUser -- bash -lc /mnt/c/Users/<you>/genrec-llm-amazon/scripts/wsl/bootstrap.sh"
exit 0
