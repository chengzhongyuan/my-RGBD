#!/bin/bash
# Run Dynamic SLAM experiment and evaluate ATE/RPE
# Paper: "基于RGB-D相机的动态场景视觉SLAM方法研究" (Wang Zhen, 2025)

set -e

ORB_EXEC="/home/czy/RGBD/ORB_SLAM2/Examples/RGB-D/rgbd_tum"
VOCAB="/home/czy/RGBD/ORB_SLAM2/Vocabulary/ORBvoc.txt"
CONFIG_FR3="/home/czy/RGBD/ORB_SLAM2/Examples/RGB-D/TUM3.yaml"
OUTPUT="/home/czy/RGBD/output"
mkdir -p $OUTPUT

DATASET=$1
NAME=$(basename $DATASET)

if [ -z "$DATASET" ]; then
    echo "Usage: $0 /path/to/tum/sequence"
    exit 1
fi

ASSOC="$DATASET/associate.txt"
MASKS="$DATASET/masks"
GT="$DATASET/groundtruth.txt"

# Generate association file if needed
if [ ! -f "$ASSOC" ]; then
    echo "Generating association file..."
    python3 -c "
import os
ds = '$DATASET'
with open(os.path.join(ds, 'rgb.txt')) as f:
    rgb = [l.strip().split() for l in f if l.strip() and not l.startswith('#')]
with open(os.path.join(ds, 'depth.txt')) as f:
    depth = [l.strip().split() for l in f if l.strip() and not l.startswith('#')]
rgb_data = [(float(r[0]), r[1]) for r in rgb]
depth_data = [(float(d[0]), d[1]) for d in depth]
with open('$ASSOC', 'w') as f:
    for rt, rf in rgb_data:
        best = min(depth_data, key=lambda x: abs(x[0] - rt))
        if abs(best[0] - rt) < 0.02:
            f.write(f'{rt:.6f} {rf} {best[0]:.6f} {best[1]}\n')
"
fi

echo "============================================"
echo "Experiment: $NAME"
echo "============================================"

# Run Dynamic SLAM (with masks if available)
TRAJ_DYN="$OUTPUT/${NAME}_dynamic.txt"
if [ -d "$MASKS" ]; then
    echo "Running Dynamic SLAM..."
    timeout 120 $ORB_EXEC $VOCAB $CONFIG_FR3 $DATASET $ASSOC $MASKS 2>&1 | tail -5
    echo "Dynamic SLAM finished."
else
    echo "No masks found at $MASKS. Run precompute_masks.py first."
fi

echo "Done. Results in $OUTPUT/"
