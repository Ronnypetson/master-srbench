#!/bin/bash
set -e

# The loader doesn't search /opt/conda/lib by default in a plain RUN shell,
# so export it explicitly for julia to find its dependent libs.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda}/lib:${LD_LIBRARY_PATH:-}"

# Sanity-check julia first: pyjulia's probe swallows julia's stderr, so if
# julia can't start we print the real error here instead of an opaque
# CalledProcessError from pysr.install().
echo "=== julia version check ==="
julia --version
echo "=== julia smoke test ==="
julia -e 'using Libdl; println("julia OK")'

# Install PySR's Julia-side dependencies into the image. Without this step
# the Python package imports, but fitting fails while initializing Julia and
# SymbolicRegression.jl.
python -c 'import pysr; pysr.install()'
