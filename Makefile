# Makefile

EXE=d2q9-bgk

CUDA_PATH=/mnt/storage/easybuild/software/CUDA/8.0.44
CC=mpiicc
CFLAGS= -cc=clang -std=c99 -Wall -O3 -Ofast -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=$(CUDA_PATH)
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
