#!/bin/bash

# Display available disk space
#echo "Checking available disk space..."
#df -h
#
## URL of the file to download (replace with your actual URL)
#URL="http://images.cocodataset.org/zips/train2017.zip"
#DOWNLOAD_DIR="/data/leo/drive1/Datasets/vis/coco"
#ZIP_FILE="$DOWNLOAD_DIR/downloaded_file.zip"
#
## Ensure the download directory exists
#mkdir -p "$DOWNLOAD_DIR"
#
## Download the file
#echo "Downloading file from $URL..."
#wget -O "$ZIP_FILE" "$URL"
#
## Unzip the downloaded file
#echo "Unzipping the file..."
#unzip "$ZIP_FILE" -d "$DOWNLOAD_DIR"
#
## Delete the ZIP file
#echo "Deleting the ZIP file..."
#rm "$ZIP_FILE"
#
## List the contents of the download directory
#echo "Contents of the download directory:"
#ls -l "$DOWNLOAD_DIR"
#
#########################################
#
## URL of the file to download (replace with your actual URL)
#URL="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json?download=true"
#DOWNLOAD_DIR="/data/leo/drive1/Datasets/vis/llava_dataset/"
#ZIP_FILE="$DOWNLOAD_DIR/llava_instruct_150k.json"
#
## Ensure the download directory exists
#mkdir -p "$DOWNLOAD_DIR"
#
## Download the file
#echo "Downloading file from $URL..."
#wget -O "$ZIP_FILE" "$URL"
#
## List the contents of the download directory
#echo "Contents of the download directory:"
#ls -l "$DOWNLOAD_DIR"

cd ~/drive1/Datasets/vis/cocostuff
echo "deleting existing files"
rm -rv train2017

unzip stuffthingmaps_trainval2017.zip

rm -r stuffthingmaps_trainval2017.zip

cd ..
rm -rv ade20k1