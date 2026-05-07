import time
import torch
def benchmark_pytorch(SIZE = 1024):
    x = torch.randn(SIZE, SIZE)
    y = torch.randn(SIZE, SIZE)

    start = time.perf_counter()
    for _ in range(100):
        z = x@y # torch.mm(x, y)
    end = time.perf_counter()
    print(f"mm avg time(torch): {(end - start) * 1000 / 100:.3f} ms")

    flops = 100 * 2 * SIZE**3 / (end - start)
    print(f"Gigaflops: {flops/1e9}")

for size in [1024, 512, 256, 128, 64]:
    print(f"Benchmarking PyTorch for size {size}x{size}")
    benchmark_pytorch(size)