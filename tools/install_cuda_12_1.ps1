param(
    [string]$InstallerUrl = "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_531.14_windows.exe",
    [string]$InstallerArgs = "-s"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This script is for Windows only."
    exit 1
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "Please run this script in an elevated PowerShell (Run as Administrator)."
}

$tempDir = Join-Path $env:TEMP "cuda-12.1.1"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

$installer = Join-Path $tempDir (Split-Path -Leaf $InstallerUrl)
Write-Host "Downloading CUDA installer..."
Invoke-WebRequest -Uri $InstallerUrl -OutFile $installer

Write-Host "Running installer..."
Start-Process -FilePath $installer -ArgumentList $InstallerArgs -Wait -NoNewWindow

if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    Write-Host "CUDA installation complete."
    nvcc --version
} else {
    Write-Host "CUDA installer finished. If nvcc is not found, log out/in or add CUDA bin to PATH."
}
