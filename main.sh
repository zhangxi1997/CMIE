#!/bin/bash
GPU=$1
MODE=$2

# for HGA+CMIE
CUDA_VISIBLE_DEVICES=$GPU python main_qa_CMIE.py --mode $MODE --bert --checkpoint HGA_CMIE

# for HGA
#CUDA_VISIBLE_DEVICES=$GPU python main_qa.py --mode $MODE --bert --checkpoint HGA

