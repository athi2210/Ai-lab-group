import matplotlib.pyplot as plt
import os

sizes = [1024, 512, 256, 128]

naive = [1.81, 1.87, 19.6, 2.53]
reordered = [28.00,27.66, 25.95, 29.54]
tiled = [22.98,25.95,26.97,29.85]
parallel = [89.22,76.62,50.31,23.83]
pytorch = [1225,1230,1075,664]

plt.figure(figsize=(9, 5))
plt.plot(sizes, naive, marker="o", label="Naive (i-j-k)")
plt.plot(sizes, reordered, marker="s", label="Reordered (i-k-j)")
plt.plot(sizes, tiled, marker="^", label="Tiled (JB=128)")
plt.plot(sizes, parallel, marker="D", label="Parallel (8 threads)")
plt.plot(sizes, pytorch, marker="x", label="PyTorch (CPU)")
plt.xlabel("Matrix Size N")
plt.ylabel("GFLOP/s")
plt.title("Matrix Multiplication Performance")
plt.xticks(sizes, [str(s) for s in sizes])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xscale("log", base=2)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/performance.png", dpi=150)
print("Saved to figures/performance.png")
plt.show()
