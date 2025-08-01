#!/usr/bin/env bash
set -e

REPO_URL="https://huggingface.co/datasets/Tex-TAR/MMTAD"
LOCAL_DIR="MMTAD"

# Clone the repo
git clone "$REPO_URL" "$LOCAL_DIR"

cd "$LOCAL_DIR"


git sparse-checkout init --cone
git sparse-checkout set test

echo "Fetched only the 'test/' split under $LOCAL_DIR/test/"