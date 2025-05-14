#!/bin/bash

# Default output directory is current directory
OUTDIR="${1:-.}"
FILENAME="causal_bootstapir_checkpoint.pt"
URL="https://storage.googleapis.com/dm-tapnet/$FILENAME"
TARGET="$OUTDIR/$FILENAME"
LINK="$OUTDIR/tapir.pt"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Download model only if it doesn't exist
if [ ! -f "$TARGET" ]; then
    echo "Downloading model to $TARGET..."
    wget -O "$TARGET" "$URL"
fi

# Create symlink if it doesn't already exist
if [ ! -L "$LINK" ]; then
    ln -s "$FILENAME" "$LINK"
fi
