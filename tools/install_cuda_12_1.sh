#!/usr/bin/env bash
set -euo pipefail

# CUDA 12.1 Linux installer (runfile). Edit variables or set env overrides if needed.
CUDA_VERSION="${CUDA_VERSION:-12.1.1}"
RUNFILE_NAME="${RUNFILE_NAME:-cuda_12.1.1_530.30.02_linux.run}"
BASE_URL="${BASE_URL:-https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers}"
INSTALL_DRIVER="${INSTALL_DRIVER:-0}" # set to 1 to install NVIDIA driver from the runfile

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux only."
  exit 1
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "WSL detected. Install CUDA on Windows for WSL instead of this script."
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Only x86_64 is supported by this script."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
  echo "sudo not found and not running as root."
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

runfile="${tmpdir}/${RUNFILE_NAME}"
url="${BASE_URL}/${RUNFILE_NAME}"

if command -v curl >/dev/null 2>&1; then
  curl -fL "$url" -o "$runfile"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$runfile" "$url"
else
  echo "curl or wget is required to download the installer."
  exit 1
fi

chmod +x "$runfile"

args=(--silent --toolkit)
if [[ "$INSTALL_DRIVER" == "1" ]]; then
  args+=(--driver)
fi

if [[ "$(id -u)" -eq 0 ]]; then
  sh "$runfile" "${args[@]}"
else
  sudo sh "$runfile" "${args[@]}"
fi

echo "CUDA toolkit installed. If nvcc is not on PATH, add:"
echo "  export PATH=/usr/local/cuda-12.1/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:\$LD_LIBRARY_PATH"
