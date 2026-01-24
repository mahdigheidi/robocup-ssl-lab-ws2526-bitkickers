#! /usr/bin/env bash

if ! command -v ./protoc-3.20/bin/protoc &> /dev/null ; then
    # brew install protobuf
    sudo apt install protobuf-compiler
fi


touch __init__.py

for proto in `ls *.proto`; do
    ./protoc-3.20/bin/protoc --python_out=. $proto
done

# in the created _pb2.py files you have to update the import path to include src.network.proto2
