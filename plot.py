import matplotlib.pyplot as plt
import os

sizes = [256, 1024]

naive    = [1.91, 1.08]
reordered = [19.38, 18.42]
tiled    = [2.72, 2.50]
parallel = [2.61, 2.53]

plt.figure(figsize=(9, 5))
plt.plot(sizes, naive,     marker='o', label='Naive (i-j-k)')
plt.plot(sizes, reordered, marker='s', label='Reordered (i-k-j)')
plt.plot(sizes, tiled,     marker='^', label='Tiled (JB=128)')
plt.plot(sizes, parallel,  marker='D', label='Parallel (4 threads)')

plt.xlabel('Matrix Size N')
plt.ylabel('GFLOP/s')
plt.title('Matrix Multiplication Performance')
plt.xticks(sizes, [str(s) for s in sizes])
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/performance.png', dpi=150)
print("Saved to figures/performance.png")
plt.show()
