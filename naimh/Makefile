# ─── Makefile for Lab 1: Matrix Multiplication on CPU ───────────────────────
CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -lm

all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul $(LDFLAGS)

clean:
	rm -f matmul *.o

benchmark: matmul
	@echo "Running benchmark..."
	./matmul

.PHONY: all clean benchmark
