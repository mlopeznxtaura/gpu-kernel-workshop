#!/bin/bash
set -e
echo "[gpu-kernel-workshop] Setting up..."
pip install --upgrade pip
pip install -r requirements.txt
python -c "import triton; print('Triton', triton.__version__)"
python -c "import cupy; print('CuPy', cupy.__version__)"
echo "Setup complete."
