"""
Graph Colouring Script
Oisin Mc Laughlin - 22441106
Ciaran Gray - 22427722
"""

import networkx as nx
import matplotlib.pyplot as plt
import random


# set random seed for reproducibility
random.seed(42)

# params
n = 50 # num nodes in graph
p = 0.15 # probability of an edge between any two nodes
num_colours = 10 # num available colours
max_steps = 500 # prevent inf loops

# build a random graph using the Erdos-Renyi model
graph = nx.erdos_renyi_graph(n, p, seed=42)

colours = {} # colours is a dict maps node id to which colour
for node in graph.nodes():
    # assign a random colour to each node at the start
    colours[node] = random.randint(0, num_colours - 1)

print("Graph created:")
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Avg degree: {sum(dict(graph.degree()).values()) / n:.2f}")


def count_conflicts(graph, colours):
    """
    Count how many edges connect nodes of the same colour
    :param graph: NetworkX graph
    :param colours: dict mapping node IDs to their assigned colour
    :return: total conflicts
    """
    conflicts = 0

    for x, y in graph.edges(): # for each edge in graph
        if colours[x] == colours[y]: # if both nodes have same colour increment conflict count
            conflicts += 1

    return conflicts


def pick_safe_colour(node, graph, colours, n_colours):
    """
    Pick a colour for the given node that is not used by any of its neighbours.
    :param node: node ID to update
    :param graph: NetworkX graph
    :param colours: dict mapping node IDs to their assigned colour
    :param n_colours: total number of available colours
    :return: colour ID to assign to the node
    """
    # collect all colours currently used by this node's neighbours
    neighbour_colours = {colours[neighbour] for neighbour in graph.neighbors(node)}

    # find colours that are not used by any neighbour
    safe_colours = [c for c in range(n_colours) if c not in neighbour_colours]

    if safe_colours:
        return random.choice(safe_colours) # pick one of the safe colours at random
    else:
        return random.randint(0, n_colours - 1) # if no safe just any random colour


def run_simulation(graph, colours, n_colours, max_steps):

    current_colours = colours.copy() # keep og colours copy

    # record how many conflicts we have at each step
    conflict_history = [count_conflicts(graph, current_colours)]

    print(f"\nStarting simulation")
    print(f"Initial conflicts: {conflict_history[0]}")

    for i in range(max_steps):
        conflicting_nodes = [] # which nodes conflict this step

        for node in graph.nodes():
            for neighbour in graph.neighbors(node):
                # check if node has conflict with neighbour
                if current_colours[node] == current_colours[neighbour]:
                    conflicting_nodes.append(node) # add to list of nodes to update (break as only need to know 1 conflict)
                    break

        # if no conflicts it is solved
        if not conflicting_nodes:
            print(f"\n\nSolved, No conflicts remaining after {i} steps.")
            break


        new_colours = {}
        for node in conflicting_nodes:
            # pick new safe colour for each conflicting node
            new_colours[node] = pick_safe_colour(node, graph, current_colours, n_colours)

        for node, new_colour in new_colours.items():
            # update the colour of the node to the new colour
            current_colours[node] = new_colour

        # count how many conflicts
        conflict_history.append(count_conflicts(graph, current_colours))

    else:
        # max iterations condition
        print(f"\n\nReached max steps ({max_steps}).")
        print(f"Final conflicts: {conflict_history[-1]}")

    return current_colours, conflict_history


# Run simulation
final_colours, conflict_history = run_simulation(graph, colours, num_colours, max_steps)


plt.figure(figsize=(10, 5))
plt.plot(conflict_history, label="Conflicts")
plt.xlabel("Time Step")
plt.ylabel("Number of Conflicts")
plt.title(f"Graph Colouring Conflicts Over Time\nN={n}, p={p}, {num_colours} colours")
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(graph, seed=42) # get pos for each node using spring layout so cleaner
node_vals = [colours[node] for node in graph.nodes()]

nodes = nx.draw_networkx_nodes(
    graph,
    pos,
    node_color=node_vals,
    cmap=plt.cm.tab10, # map colour IDs to colours
    node_size=300,
)
nx.draw_networkx_edges(graph, pos, alpha=0.5)
nx.draw_networkx_labels(graph, pos)

cbar = plt.colorbar(nodes)
cbar.set_label("Colour ID")
if num_colours > 1:
    nodes.set_clim(0, num_colours - 1) # set colour limits for colourbar

plt.title("Final Graph Colouring")
plt.axis("off")
plt.tight_layout()
plt.show()
