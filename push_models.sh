#!/bin/bash
# NOTE: first makesure predict.py is set to mamba-130m
# then run this script

models=(mamba-130m mamba-370m mamba-790m mamba-1.4b mamba-2.8b mamba-2.8b-slimpj)

for t in ${models[@]}; do
    if [ -e mamba-model-cache ]; then
        sudo rm -r mamba-model-cache
    fi
    if [ -n "$prev" ]; then
        sed "0,/$prev/ s//$t/" predict.py > predict.py.tmp && mv predict.py.tmp predict.py
    fi
    cog predict -i prompt="genel" && cog build && cog push r8.im/adirik/$t > $t.log
    prev=$t
done
