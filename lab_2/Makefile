# ─── Makefile for Lab 1: Matrix Multiplication on CPU ───────────────────────
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp-simd -pthread
LDFLAGS = -lm -pthread

# Targets
all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul $(LDFLAGS)

clean:
	rm -f matmul *.o

pytorch:
	python benchmark_pytorch.py

benchmark: matmul
	@echo "Running comprehensive benchmark..."
	./matmul

.PHONY: all clean run_matmul run_ml benchmark
