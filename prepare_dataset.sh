#!/bin/bash

# Default download directory is current directory or specified as first argument
DOWNLOAD_DIR="${1:-.}"
DATASET_URL="https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip"
ZIP_FILE="$DOWNLOAD_DIR/tapvid_davis.zip"
UNZIPPED_DIR="$DOWNLOAD_DIR/tapvid_davis"
CHECK_FILE="$UNZIPPED_DIR/tapvid_davis.pkl"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Exit if the target file already exists
if [ -f "$CHECK_FILE" ]; then
  exit 0
fi

# Download the zip file
echo "Downloading dataset to $ZIP_FILE..."
wget -O "$ZIP_FILE" "$DATASET_URL" || { echo "Download failed"; exit 1; }

# Unzip the dataset
echo "Unzipping dataset..."
unzip -q "$ZIP_FILE" -d "$DOWNLOAD_DIR" || { echo "Unzip failed"; exit 1; }

# Remove the zip file
echo "Cleaning up..."
rm -f "$ZIP_FILE"


