#!/bin/bash
echo ">>> Configuring SSH to disable Host Key Checking..."
# This prevents the benchmark from hanging on "Are you sure you want to connect?"
echo "Host *" > ~/.ssh/config
echo "    StrictHostKeyChecking no" >> ~/.ssh/config
chmod 600 ~/.ssh/config

echo ">>> Generating SSH Key (Press Enter through prompts)..."
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -N "" -q
    echo "Key generated."
else
    echo "Key already exists."
fi

echo "=========================================================="
echo "SSH CONFIG DONE."
echo "MANUAL STEP REQUIRED:"
echo "1. Run: cat ~/.ssh/id_rsa.pub"
echo "2. Copy the output."
echo "3. Paste it into ~/.ssh/authorized_keys on the OTHER node."
echo "=========================================================="