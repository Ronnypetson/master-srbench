#!/bin/bash
set -e

# Sanity-check julia first: pyjulia's probe swallows julia's stderr, so if
# julia can't even start (e.g. libgit2/crypto issues) we print the real error
# here instead of an opaque CalledProcessError from pysr.install().
echo "julia version check:"
julia --version
echo "julia smoke test:"
julia -e 'using Libdl; println("julia OK")'

# Install PySR's Julia-side dependencies into the image. Without this step
# the Python package imports, but fitting fails while initializing Julia and
# SymbolicRegression.jl.
python -c 'import pysr; pysr.install()'
