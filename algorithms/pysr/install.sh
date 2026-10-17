#!/bin/bash
set -e

# conda-forge julia needs libutf8proc.so.2 (from the utf8proc package) and
# friends at runtime. The loader doesn't search /opt/conda/lib by default in
# a plain RUN shell, so export it explicitly for this script.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda}/lib:${LD_LIBRARY_PATH:-}"

echo "=== julia library diagnostics ==="
JULIA_LIBDIR="$(dirname "$(readlink -f "$(which julia)")")/../lib"
echo "ldd libjulia-internal:"
ldd "${JULIA_LIBDIR}/julia/libjulia-internal.so."* || true
echo "utf8proc in conda lib:"
ls -la "${CONDA_PREFIX:-/opt/conda}/lib/" | grep -i utf8proc || echo "NOT FOUND"

# If utf8proc is missing, install it explicitly (it should be a julia dep,
# but solver/base-image differences can drop it).
if ! ls "${CONDA_PREFIX:-/opt/conda}/lib/"libutf8proc.so* >/dev/null 2>&1; then
    echo "utf8proc missing — installing explicitly"
    micromamba install -n base -y -c conda-forge utf8proc
fi

echo "=== julia version check ==="
julia --version
echo "=== julia smoke test ==="
julia -e 'using Libdl; println("julia OK")'

# Install PySR's Julia-side dependencies into the image. Without this step
# the Python package imports, but fitting fails while initializing Julia and
# SymbolicRegression.jl.
python -c 'import pysr; pysr.install()'
