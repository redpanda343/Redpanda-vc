#!/usr/bin/env bash
set -euo pipefail

echo "Welcome to the Applio Installer!"
echo

INSTALL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_DIR="$INSTALL_DIR/env"
CONDA_EXE="$MINICONDA_DIR/bin/conda"

case "$(uname -m)" in
    x86_64)         MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_25.11.1-1-Linux-x86_64.sh" ;;
    aarch64|arm64)  MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_25.11.1-1-Linux-aarch64.sh" ;;
    *)              echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

start_time=$SECONDS

error() {
    echo "An error occurred during installation. Please check the output above for details."
    exit 1
}
trap error ERR

install_miniconda() {
    if [ -x "$CONDA_EXE" ]; then
        echo "Miniconda already installed. Skipping installation."
        return 0
    fi

    echo "Miniconda not found. Starting download and installation..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$MINICONDA_URL" -o miniconda.sh
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$MINICONDA_URL" -O miniconda.sh
    else
        echo "Neither curl nor wget is available. Please install one of them and try again."
        exit 1
    fi

    if [ ! -f miniconda.sh ]; then
        echo "Download failed. Please check your internet connection and try again."
        exit 1
    fi

    bash miniconda.sh -b -p "$MINICONDA_DIR" || { echo "Miniconda installation failed."; exit 1; }
    rm -f miniconda.sh
    echo "Miniconda installation complete."
    echo
}

accept_channel_tos() {
    # Newer conda refuses to install from the default Anaconda channels until
    # their Terms of Service have been accepted.
    if "$CONDA_EXE" tos --help >/dev/null 2>&1; then
        echo "Accepting Terms of Service for the default Anaconda channels..."
        for channel in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
            "$CONDA_EXE" tos accept --override-channels --channel "$channel" >/dev/null 2>&1 || true
        done
        echo
    fi
}

create_conda_env() {
    echo "Creating Conda environment..."
    "$CONDA_EXE" create -y --prefix "$ENV_DIR" python=3.12
    echo "Conda environment created successfully."
    echo

    if [ -x "$ENV_DIR/bin/python" ]; then
        echo "Installing uv package installer..."
        "$ENV_DIR/bin/python" -m pip install uv
        echo "uv installation complete."
        echo
    fi
}

install_dependencies() {
    echo "Installing dependencies..."
    # shellcheck disable=SC1091
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"
    uv pip uninstall onnxruntime
    uv pip install -r "$INSTALL_DIR/requirements.txt" \
        --reinstall-package onnxruntime-gpu \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match
    uv pip install --no-deps swift-f0==0.3.0
    conda deactivate
    echo "Dependencies installation complete."
    echo
}

install_miniconda
accept_channel_tos
create_conda_env
install_dependencies

elapsed=$((SECONDS - start_time))
printf 'Installation time: %d hours, %d minutes, %d seconds.\n\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))

echo "Applio has been installed successfully!"
echo "To start Applio, please run './run-applio.sh'."
echo
