#!/bin/bash
for arch in "NN" "CNN" 
do 
    python test.py -a $arch
done