#!/bin/bash
echo "Starting setup script..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
echo "Installed OpenGL dependencies."
pip install -r requirements.txt
echo "Installed Python dependencies."
