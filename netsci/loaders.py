"""Dataset loading utilities for the Network Science Lab Course.

Provides a simple API for loading pre-bundled network datasets from the
``data/`` directory. All datasets are stored as GraphML files and require
no runtime downloads -- everything works offline.

Available datasets::

    >>> from netsci.loaders import list_graphs
    >>> list_graphs()
    ['airports', 'arxiv', 'email', 'facebook', 'football', 'got', 'karate', 'lesmis', 'powergrid', 'protein']

Load any dataset by name::

    >>> from netsci.loaders import load_graph
    >>> G = load_graph("karate")
    karate: 34 nodes, 78 edges (undirected)

Datasets
--------
- **karate** -- Zachary Karate Club (34 nodes, 78 edges, undirected)
- **lesmis** -- Les Miserables co-appearance (77 nodes, 254 edges, undirected)
- **powergrid** -- US Power Grid (4941 nodes, 6594 edges, undirected)
- **facebook** -- Facebook ego network 0 (334 nodes, 2852 edges, undirected)
- **arxiv** -- Arxiv GR-QC co-authorship (5242 nodes, 14496 edges, undirected)
- **airports** -- US Top-500 airport routes (500 nodes, 2980 edges, undirected)
- **email** -- EU email communication (1005 nodes, 25571 edges, directed)
- **football** -- College football conferences (115 nodes, 613 edges, undirected)
- **got** -- Game of Thrones interactions (796 nodes, 2823 edges, undirected)
- **protein** -- Yeast protein interactions (1870 nodes, 2277 edges, undirected)
"""

from pathlib import Path

import networkx as nx

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# name -> (subdir, filename, node_type, directed)
_REGISTRY = {
    "karate":    ("week1", "karate.graphml",    int,  False),
    "lesmis":    ("week1", "lesmis.graphml",    str,  False),
    "powergrid": ("week2", "powergrid.graphml", int,  False),
    "facebook":  ("week2", "facebook.graphml",  int,  False),
    "arxiv":     ("week2", "arxiv_grqc.graphml", int, False),
    "airports":  ("week3", "airports.graphml",  str,  False),
    "email":     ("week3", "email.graphml",     int,  True),
    "football":  ("week5", "football.graphml",  str,  False),
    "got":       ("week5", "got.graphml",       str,  False),
    "protein":   ("week2", "protein.graphml",  int,  False),
}


def list_graphs() -> list[str]:
    """Return sorted list of available dataset names.

    Returns
    -------
    list[str]
        Alphabetically sorted dataset names that can be passed to
        :func:`load_graph`.
    """
    return sorted(_REGISTRY.keys())


def load_graph(name: str) -> nx.Graph | nx.DiGraph:
    """Load a named dataset and return a NetworkX graph.

    Parameters
    ----------
    name : str
        Short name of the dataset (e.g. ``"karate"``, ``"email"``).
        Use :func:`list_graphs` to see all available names.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        The loaded graph. Directed datasets (e.g. email) return
        ``nx.DiGraph``; all others return ``nx.Graph``.

    Raises
    ------
    ValueError
        If *name* is not a recognized dataset. The error message lists
        all available dataset names.
    """
    if name not in _REGISTRY:
        available = ", ".join(list_graphs())
        raise ValueError(
            f"Unknown graph '{name}'. Available: {available}"
        )

    subdir, filename, node_type, directed = _REGISTRY[name]
    path = _DATA_DIR / subdir / filename
    G = nx.read_graphml(path, node_type=node_type)

    direction = "directed" if G.is_directed() else "undirected"
    print(f"{name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({direction})")

    return G
