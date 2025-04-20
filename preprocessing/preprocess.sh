#!/bin/bash

echo "Starting augmentation script..."
python3 /home/slsecret/TrafficObjectDetection/preprocessing/apply_augmentations.py
if [ $? -ne 0 ]; then
    echo "script failed. Exiting."
    exit 1
fi

echo "Starting resize script..."
python3 /home/slsecret/TrafficObjectDetection/preprocessing/resize_normal.py
if [ $? -ne 0 ]; then
    echo "script failed. Exiting."
    exit 1
fi

echo "Starting create dataset a script..."
python3 /home/slsecret/TrafficObjectDetection/preprocessing/create_datasets_a.py
if [ $? -ne 0 ]; then
    echo "failed. Exiting."
    exit 1
fi

echo "Starting create dataset c script..."
python3 /home/slsecret/TrafficObjectDetection/preprocessing/create_datasets_c.py
if [ $? -ne 0 ]; then
    echo "failed. Exiting."
    exit 1
fi

echo "Starting combine dataset script..."
python3 /home/slsecret/TrafficObjectDetection/preprocessing/combine_datasets.py
if [ $? -ne 0 ]; then
    echo "failed. Exiting."
    exit 1
fi

echo "All scripts completed successfully."
