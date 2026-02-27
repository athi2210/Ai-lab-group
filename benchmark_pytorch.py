import time
import torch

SIZE = 1024
x = torch.randn(SIZE, SIZE)
y = torch.randn(SIZE, SIZE)

start = time.perf_counter()
for _ in range(100):
    z = torch.mm(x, y)
end = time.perf_counter()
print(f"mm avg time(torch): {(end - start) * 1000 / 100:.3f} ms")