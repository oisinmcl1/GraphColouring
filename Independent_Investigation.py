import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from Baseline_Graph_Colouring import count_conflicts
from Baseline_Graph_Colouring import pick_safe_colour

# set random seed for reproducibility
random.seed(42)

# params
n = 50 # num nodes in graph
p = 0.15 # probability of an edge between any two nodes
num_colours = 10 # num available colours
max_steps = 500 # prevent inf loops

graphs_to_test = {}

# 1. Ring graph
graphs_to_test["Ring"] = nx.cycle_graph(n)

# 2. 2D Grid (7x7 = 49 nodes, closest to 50)
grid = nx.grid_2d_graph(7, 7)
graphs_to_test["2D Grid (7x7)"] = nx.convert_node_labels_to_integers(grid)

# 3. Erdos-Renyi (same as step 1)
graphs_to_test["Erdos-Renyi"] = nx.erdos_renyi_graph(n, p, seed=42)

# 4. Barabasi-Albert scale-free
graphs_to_test["Barabasi-Albert"] = nx.barabasi_albert_graph(n, 3, seed=42)


def get_chromatic_number(g):
    """Approximate the chromatic number using greedy colouring."""
    colouring = nx.coloring.greedy_color(g, strategy="largest_first")
    return max(colouring.values()) + 1


def run_trial(g, n_colours, max_steps, seed=None):
    """
    Run one trial of the decentralised colouring algorithm on any graph.
    Returns (success, steps_taken, final_conflicts, conflict_history)
    """
    if seed is not None:
        random.seed(seed)

    # random initial colouring
    trial_colours = {node: random.randint(0, n_colours - 1) for node in g.nodes()}
    history = [count_conflicts(g, trial_colours)]

    for step in range(max_steps):
        conflicting_nodes = []
        for node in g.nodes():
            for neighbour in g.neighbors(node):
                if trial_colours[node] == trial_colours[neighbour]:
                    conflicting_nodes.append(node)
                    break

        if not conflicting_nodes:
            return True, step, 0, history

        # synchronous update
        new_colours = {}
        for node in conflicting_nodes:
            new_colours[node] = pick_safe_colour(node, g, trial_colours, n_colours)

        for node, new_colour in new_colours.items():
            trial_colours[node] = new_colour

        history.append(count_conflicts(g, trial_colours))

    return False, max_steps, history[-1], history


# --- Run the experiment across all topologies ---

n_trials = 30
results = {}

for name, g in graphs_to_test.items():
    chromatic_number = get_chromatic_number(g)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    max_deg = max(dict(g.degree()).values())
    avg_deg = sum(dict(g.degree()).values()) / n_nodes

    print(f"\n{'=' * 60}")
    print(f"Graph: {name}")
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"  Avg degree: {avg_deg:.2f}, Max degree: {max_deg}")
    print(f"  Chromatic number (greedy): {chromatic_number}")
    print(f"{'=' * 60}")

    k_min = max(chromatic_number - 1, 2)
    k_max = chromatic_number + 8
    colour_range = list(range(k_min, k_max + 1))

    graph_results = {
        "chromatic_number": chromatic_number,
        "max_degree": max_deg,
        "colour_range": colour_range,
        "success_rates": [],
        "avg_convergence_times": [],
    }

    for k in colour_range:
        successes = 0
        convergence_times = []

        for trial in range(n_trials):
            seed = trial * 1000 + k
            success, steps, conflicts, _ = run_trial(g, k, max_steps, seed=seed)

            if success:
                successes += 1
                convergence_times.append(steps)

        success_rate = successes / n_trials * 100
        avg_time = np.mean(convergence_times) if convergence_times else float("nan")

        graph_results["success_rates"].append(success_rate)
        graph_results["avg_convergence_times"].append(avg_time)

        print(f"  k={k:2d} (chromatic_number{k - chromatic_number:+d}): "
              f"success={success_rate:6.1f}%, "
              f"avg_steps={avg_time:6.1f}")

    results[name] = graph_results

# --- Plot 1: Success rate vs colour headroom ---

plt.figure(figsize=(10, 6))
for name, data in results.items():
    chromatic_number = data["chromatic_number"]
    x = [k - chromatic_number for k in data["colour_range"]]
    plt.plot(x, data["success_rates"], marker="o", linewidth=2,
             label=f'{name} (χ={chromatic_number})')

plt.xlabel("Colours Above Chromatic Number (k - χ)")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate vs Colour Headroom by Topology")
plt.ylim(-5, 105)
plt.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.3, label="χ (chromatic number)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("success_rate_vs_headroom.png", dpi=150)
plt.show()

# --- Plot 2: Colour headroom vs convergence time
plt.figure(figsize=(10, 6))
x_cap = 30  # cap x-axis so the Ring outlier at 271 steps doesn't squash everything

for name, data in results.items():
    chi = data["chromatic_number"]
    y = [k - chi for k in data["colour_range"]]
    times = data["avg_convergence_times"]

    # separate plottable points from outliers beyond the cap
    plot_times = []
    plot_y = []
    for t, yi in zip(times, y):
        if not np.isnan(t) and t <= x_cap:
            plot_times.append(t)
            plot_y.append(yi)
        elif not np.isnan(t) and t > x_cap:
            # annotate the outlier at the edge of the graph
            plt.annotate(f'{name}: {t:.0f} steps',
                         xy=(x_cap, yi), fontsize=9,
                         ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.plot(plot_times, plot_y, marker="s", linewidth=2,
             label=f'{name} (χ={chi})')

plt.xlabel("Average Convergence Time (steps)")
plt.ylabel("Colours Above Chromatic Number (k - χ)")
plt.title("Colour Headroom vs Convergence Speed by Topology")
plt.xlim(0, x_cap)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.3, label="χ (chromatic number)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence_time_vs_headroom.png", dpi=150)
plt.show()

print(f"\n------------")
print("\nSUMMARY: Minimum colours for 100% success rate (30 trials)")
print(f"\n------------")
print(f"{'Graph':<20} {'χ':>4} {'Min k (100%)':>14} {'Headroom':>10} {'Max Degree':>12}")
print(f"\n------------")

for name, data in results.items():
    chromatic_number = data["chromatic_number"]
    min_k_full = None
    for k, rate in zip(data["colour_range"], data["success_rates"]):
        if rate == 100.0:
            min_k_full = k
            break

    headroom = f"+{min_k_full - chromatic_number}" if min_k_full else "never"
    min_k_str = str(min_k_full) if min_k_full else "N/A"
    print(f"{name:<20} {chromatic_number:>4} {min_k_str:>14} {headroom:>10} {data['max_degree']:>12}")
print(f"\n------------")
