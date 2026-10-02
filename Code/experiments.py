import time
import matplotlib.pyplot as plt
from ba_core import blahut_arimoto
from ba_accelerated import accelerated_ba
from channels import BSC, BEC

def run_experiment(W, name):
    t0 = time.time()
    res_std = blahut_arimoto(W)
    t1 = time.time()

    res_acc = accelerated_ba(W)
    t2 = time.time()

    print(f"\n{name}")
    print("Standard BA:")
    print(f"  Capacity: {res_std['capacity']:.6f} bits")
    print(f"  Iterations: {res_std['iterations']}")
    print(f"  Time: {t1 - t0:.3f}s")

    print("Accelerated BA:")
    print(f"  Capacity: {res_acc['capacity']:.6f} bits")
    print(f"  Iterations: {res_acc['iterations']}")
    print(f"  Time: {t2 - t1:.3f}s")

    return res_std, res_acc

def plot_convergence(std, acc, title):
    plt.figure()
    plt.plot(std["history"], label="Standard BA")
    plt.plot(acc["history"], label="Accelerated BA")
    plt.xlabel("Iteration")
    plt.ylabel("Capacity (bits)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    W_bsc = BSC(0.1)
    std, acc = run_experiment(W_bsc, "BSC(p=0.1)")
    plot_convergence(std, acc, "BSC Convergence")

    W_bec = BEC(0.2)
    std, acc = run_experiment(W_bec, "BEC(e=0.2)")
    plot_convergence(std, acc, "BEC Convergence")
