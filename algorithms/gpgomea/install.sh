set -euxo pipefail

# remove directory if it exists
if [ -d "MultiGPG" ]; then
    rm -rf MultiGPG
fi

git clone https://github.com/matigekunstintelligentie/MultiGPG.git

cd MultiGPG || exit
# ! don't forget to update this
git checkout dd6954fe53e0532101789db7db7951d95231931e

# Upstream compiles Release builds with -march=native, which bakes the build
# machine's ISA (e.g. AVX2/AVX-512 on GH Actions runners) into _pb_mgpg.so.
# The image then crashes with SIGILL ("Illegal instruction") on import on any
# machine whose CPU lacks those instructions. Target a portable baseline
# instead (x86-64-v2 ~= any x86_64 CPU since ~2009; use plain x86-64 for
# maximum compatibility, or x86-64-v3 if all targets have AVX2).
sed -i 's/-march=native/-march=x86-64-v2 -mtune=generic/' CMakeLists.txt

# doesn't seem to work and adds bloat...
# make

# 1. build .so
mkdir -p build
cd build || exit
cmake -S ../ -B . -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) -DCMAKE_BUILD_TYPE=Release
make

# build python package
cd .. || exit
cp -r src/pypkg ..
mv build/_pb_mgpg*.so ../pypkg/pymgpg/_pb_mgpg.so
cd ../pypkg
python setup.py install

# cleanup
cd .. || exit
rm -rf MultiGPG
