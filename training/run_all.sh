#!/bin/bash

echo "Starting training script..."
python3 /home/slsecret/TrafficObjectDetection/training/train.py
if [ $? -ne 0 ]; then
    echo "Training script failed. Exiting."
    exit 1
fi

echo "Starting test_all_epochs script..."
python3 /home/slsecret/TrafficObjectDetection/training/test_all_epochs.py
if [ $? -ne 0 ]; then
    echo "Testing script failed. Exiting."
    exit 1
fi

echo "Starting Rainbow DQN script..."
python3 /home/slsecret/COMP579/project/src/rainbow_dqn/rainbow_dqn_color.py
if [ $? -ne 0 ]; then
    echo "Rainbow DQN script failed. Exiting."
    exit 1
fi

echo "All scripts completed successfully."
