"""Utility functions for the Network Science Lab Course.

Provides:
  - ``SEED`` -- fixed random seed (42) for reproducible results across all
    notebooks.  Use when calling any function that accepts a ``seed``
    parameter (e.g., ``nx.community.louvain_communities(G, seed=SEED)``).
  - ``graph_summary(G)`` -- prints a concise multi-line statistical
    overview of a NetworkX graph.
"""

import networkx as nx
import numpy as np

# Fixed random seed for reproducible results across all notebooks.
# Use this constant when calling functions that accept a seed parameter
# (e.g., nx.spring_layout(G, seed=SEED)).
SEED = 42


def graph_summary(G):
    """Print a concise statistical overview of graph *G*.

    Displays node/edge counts, density, average degree, number of
    connected components, and (for undirected graphs) average clustering
    coefficient.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Any NetworkX graph.

    Returns
    -------
    None
        Output is printed to stdout.
    """
    name = G.name if G.name else "Graph"
    graph_type = "directed" if G.is_directed() else "undirected"
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = sum(d for _, d in G.degree()) / max(n_nodes, 1)

    if G.is_directed():
        n_components = nx.number_weakly_connected_components(G)
    else:
        n_components = nx.number_connected_components(G)

    print(f"Graph Summary: {name}")
    print(f"  Type:        {graph_type}")
    print(f"  Nodes:       {n_nodes}")
    print(f"  Edges:       {n_edges}")
    print(f"  Density:     {density:.4f}")
    print(f"  Avg degree:  {avg_degree:.2f}")
    print(f"  Components:  {n_components}")

    if not G.is_directed():
        avg_clust = nx.average_clustering(G)
        print(f"  Avg clustering: {avg_clust:.4f}")


def small_world_table(G, name, n_rand=5):
    """Print small-world statistics for graph *G* vs a random ER baseline.

    Computes clustering coefficient and average path length for the real
    graph, generates *n_rand* Erdos-Renyi graphs with the same density,
    and prints the small-world coefficient sigma.

    Parameters
    ----------
    G : networkx.Graph
        An undirected, connected graph.
    name : str
        Label used in the printed output.
    n_rand : int, default 5
        Number of random baseline graphs to average over.

    Returns
    -------
    None
        Output is printed to stdout.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    C_real = nx.average_clustering(G)
    L_real = nx.average_shortest_path_length(G)

    p_er = 2 * m / (n * (n - 1))
    rng = np.random.default_rng(SEED)
    Cs, Ls = [], []
    for _ in range(n_rand):
        Gr = nx.erdos_renyi_graph(n, p_er, seed=int(rng.integers(1e6)))
        if nx.is_connected(Gr):
            Cs.append(nx.average_clustering(Gr))
            Ls.append(nx.average_shortest_path_length(Gr))
    C_rand, L_rand = np.mean(Cs), np.mean(Ls)
    sigma = (C_real / C_rand) / (L_real / L_rand)

    print(f"\n{name}")
    print(f"  C_real={C_real:.4f}  C_rand={C_rand:.4f}  ratio={C_real/C_rand:.1f}x")
    print(f"  L_real={L_real:.2f}    L_rand={L_rand:.2f}    ratio={L_real/L_rand:.2f}")
    print(f"  sigma = {sigma:.2f}  {'<-- small world!' if sigma > 1 else ''}")


def fit_power_law(degrees, k_min=1):
    """Estimate power-law exponent using Maximum Likelihood Estimation.

    Parameters
    ----------
    degrees : array-like
        Observed degree values.
    k_min : int, default 1
        Minimum degree threshold.  Only values ``>= k_min`` are used.

    Returns
    -------
    float
        Estimated exponent alpha.
    """
    degrees = np.array([d for d in degrees if d >= k_min])
    n = len(degrees)
    alpha = 1 + n / np.sum(np.log(degrees / k_min))
    return alpha


def partition_to_labels(G, partition):
    """Convert a community partition to a list of integer labels.

    Parameters
    ----------
    G : networkx.Graph
        The graph whose nodes define the label order.
    partition : list of sets
        Each set contains the nodes belonging to one community.

    Returns
    -------
    list[int]
        Integer label for each node, ordered as ``G.nodes()``.
    """
    node_to_label = {}
    for i, comm in enumerate(partition):
        for n in comm:
            node_to_label[n] = i
    return [node_to_label[n] for n in G.nodes()]
