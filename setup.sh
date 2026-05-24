#!/bin/bash
# Setup script for Dynamic SLAM project
# Downloads ORB vocabulary and builds Pangolin + ORB-SLAM2

set -e

echo "=== Downloading ORB Vocabulary ==="
if [ ! -f ORB_SLAM2/Vocabulary/ORBvoc.txt ]; then
    cd ORB_SLAM2/Vocabulary
    wget https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz
    tar -xzf ORBvoc.txt.tar.gz
    rm ORBvoc.txt.tar.gz
    cd ../..
    echo "Vocabulary downloaded."
else
    echo "Vocabulary already exists."
fi

echo "=== Building Thirdparty libraries ==="
cd ORB_SLAM2/Thirdparty/DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

cd ../../g2o
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../../..

echo "=== Building ORB-SLAM2 ==="
cd ORB_SLAM2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/.local"
make -j$(nproc)
cd ../..

echo "=== Setup complete! ==="
