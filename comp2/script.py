import numpy as np
import time
import tracemalloc
import random
import sys

# -----------------------------
# Config
# -----------------------------
N = 100000   # number of samples
D = 200    # number of features


# -----------------------------
# NumPy version
# -----------------------------
def numpy_linear_regression(X, w):
    return X @ w


# -----------------------------
# Pure Python version
# -----------------------------
def python_linear_regression(X, w):
    result = []
    for row in X:
        s = 0.0
        for i in range(len(w)):
            s += row[i] * w[i]
        result.append(s)
    return result


# -----------------------------
# Data generation
# -----------------------------
print("Generating data...")

X_numpy = np.random.rand(N, D)
w_numpy = np.random.rand(D)

X_list = X_numpy.tolist()
w_list = w_numpy.tolist()


# -----------------------------
# Benchmark NumPy
# -----------------------------
tracemalloc.start()

start = time.perf_counter()
y_numpy = numpy_linear_regression(X_numpy, w_numpy)
end = time.perf_counter()

numpy_current, numpy_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

numpy_time = end - start


# -----------------------------
# Benchmark Python lists
# -----------------------------
tracemalloc.start()

start = time.perf_counter()
y_python = python_linear_regression(X_list, w_list)
end = time.perf_counter()

python_current, python_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

python_time = end - start


# -----------------------------
# Results
# -----------------------------
print("\n--- Benchmark Results ---")

print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"Python list time: {python_time:.6f} seconds")

print(f"\nNumPy peak memory: {numpy_peak / 1024 / 1024:.4f} MB")
print(f"Python peak memory: {python_peak / 1024 / 1024:.4f} MB")

speedup = python_time / numpy_time
print(f"\nNumPy speedup: {speedup:.2f}x")