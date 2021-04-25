#!/bin/bash
module load libraries/cuda
make clean &>/dev/null
make &>/dev/null
for i in {1..10}
do
    python bench.py >>rezultate
done