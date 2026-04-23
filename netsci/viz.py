"""Visualization helpers for the Network Science Lab Course.

Three opinionated, one-call visualization functions for common network
analysis plots.  Each function creates its own figure, applies the
``seaborn-v0_8-muted`` style for clean academic aesthetics, and calls
``plt.show()`` -- so a single function call in a notebook cell produces
a complete, publication-ready visualization.

Design philosophy:
  - **Standalone functions only** -- no ``ax`` parameter.  Students who
    need multi-panel figures compose them directly with ``plt.subplot()``.
  - **Sensible defaults** -- one call with just a graph ``G`` gives a
    useful plot; keyword arguments allow customization when needed.
  - **Clean academic style** -- white background, muted colors, minimal
    gridlines via the ``seaborn-v0_8-muted`` matplotlib style context.

Functions
---------
plot_degree_dist(G, title=None, log=False)
    Histogram of degree values, or log-log scatter for power-law detection.
plot_ccdf(G, title=None, fit_line=False)
    Complementary CDF of the degree distribution on log-log axes.
draw_graph(G, node_color=None, node_size=None, title=None, layout="spring")
    Spring-layout node-link diagram with muted colors and optional labels.
plot_adjacency(G, title=None, cmap="Blues")
    Color-coded heatmap of the adjacency matrix.
draw_pyvis(G, node_color=None, title=None, height="500px", filename="graph.html")
    Interactive PyVis graph visualization saved to HTML and displayed inline.
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from netsci.utils import SEED


# ---------------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------------

def plot_degree_dist(G, title=None, log=False):
    """Plot the degree distribution of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    log : bool, default False
        If *True*, produce a log-log scatter plot (useful for power-law
        detection).  Otherwise produce a histogram.
    """
    degrees = [d for _, d in G.degree()]

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(6, 4))

        if log:
            deg_count = Counter(degrees)
            x, y = zip(*sorted(deg_count.items()))
            ax.scatter(x, y, s=30, alpha=0.7,
                       edgecolors="white", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title or "Degree Distribution (log-log)")
        else:
            ax.hist(degrees,
                    bins=range(min(degrees), max(degrees) + 2),
                    edgecolor="white", alpha=0.8)
            ax.set_title(title or "Degree Distribution")

        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Complementary CDF
# ---------------------------------------------------------------------------

def plot_ccdf(G, title=None, fit_line=False):
    """Plot the complementary CDF of the degree distribution on log-log axes.

    The CCDF shows P(K >= k) for each degree value k.  On log-log axes a
    power-law distribution appears as a straight line with slope -(gamma-1).

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    fit_line : bool, default False
        If *True*, overlay an MLE power-law fit line.  The exponent is
        estimated as alpha = 1 + n / sum(ln(x_i / x_min)).
    """
    degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    n = len(degrees)
    ccdf_y = np.arange(1, n + 1) / n
    ccdf_x = np.array(degrees)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ccdf_x, ccdf_y, s=20, alpha=0.7,
                   edgecolors="white", linewidth=0.5)

        if fit_line:
            k_min = min(degrees)
            log_ratios = np.log(np.array(degrees) / k_min)
            alpha = 1 + n / log_ratios.sum()
            k_range = np.logspace(np.log10(k_min), np.log10(max(degrees)), 200)
            fit_y = (k_range / k_min) ** (-(alpha - 1))
            ax.plot(k_range, fit_y, "r--", lw=1.5,
                    label=f"MLE fit  α = {alpha:.2f}")
            ax.legend(fontsize=9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("P(K ≥ k)")
        ax.set_title(title or "Degree CCDF (log-log)")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Node-link graph drawing
# ---------------------------------------------------------------------------

def draw_graph(G, node_color=None, node_size=None, title=None,
               layout="spring"):
    """Draw a node-link diagram of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    node_color : color or list of colors, optional
        Node fill color(s).  Defaults to muted blue ``"#4878CF"``.
    node_size : int or list of ints, optional
        Node marker size(s).  Defaults to 80.
    title : str, optional
        Custom plot title.
    layout : {"spring", "kamada_kawai", "circular"}, default "spring"
        Layout algorithm.  ``"spring"`` uses a deterministic seed for
        reproducibility.
    """
    if G.number_of_nodes() > 500:
        print(
            f"Warning: graph has {G.number_of_nodes()} nodes. "
            "Drawing may be slow or unreadable."
        )

    layout_fns = {
        "spring": lambda: nx.spring_layout(G, seed=SEED),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
        "circular": lambda: nx.circular_layout(G),
    }
    pos = layout_fns.get(layout, layout_fns["spring"])()

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx(
            G, pos, ax=ax,
            node_color=node_color or "#4878CF",
            node_size=node_size or 80,
            edge_color="#cccccc",
            width=0.5,
            with_labels=(G.number_of_nodes() <= 50),
            font_size=8,
            alpha=0.9,
        )
        ax.set_title(title or "")
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Adjacency matrix heatmap
# ---------------------------------------------------------------------------

def plot_adjacency(G, title=None, cmap="Blues"):
    """Plot a heatmap of the adjacency matrix of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    cmap : str, default "Blues"
        Matplotlib/seaborn colormap name.
    """
    A = nx.to_numpy_array(G)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            A, ax=ax, cmap=cmap, square=True,
            cbar_kws={"shrink": 0.8},
            xticklabels=False, yticklabels=False,
        )
        ax.set_title(title or "Adjacency Matrix")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Interactive PyVis visualization
# ---------------------------------------------------------------------------

# Default color palette for community coloring (10 distinct colors).
_PYVIS_COLORS = [
    "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66",
    "#77BEDB", "#E8A06B", "#8C8C8C", "#C572D1", "#64B5CD",
]


def draw_pyvis(G, node_color=None, title=None, height="500px",
               filename="graph.html"):
    """Render an interactive graph with PyVis and display inline.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    node_color : dict or list, optional
        Per-node colors.  A *dict* mapping ``node -> color_string`` or
        ``node -> int`` (community label mapped to a built-in palette).
        A *list* of color strings in ``G.nodes()`` order also works.
        When values are integers they are mapped to a 10-color palette.
    title : str, optional
        HTML heading displayed above the graph.
    height : str, default "500px"
        CSS height of the canvas.
    filename : str, default "graph.html"
        Path for the generated HTML file.
    """
    from pyvis.network import Network
    from IPython.display import display, HTML

    net = Network(height=height, width="100%", notebook=True,
                  cdn_resources="in_line")
    if title:
        net.heading = title

    # Build color mapping ------------------------------------------------
    nodes = list(G.nodes())
    if node_color is None:
        color_map = {n: _PYVIS_COLORS[0] for n in nodes}
    elif isinstance(node_color, dict):
        color_map = {}
        for n in nodes:
            c = node_color.get(n, 0)
            if isinstance(c, (int, np.integer)):
                color_map[n] = _PYVIS_COLORS[int(c) % len(_PYVIS_COLORS)]
            else:
                color_map[n] = c
    else:
        # list / array — same order as G.nodes()
        color_map = {n: (_PYVIS_COLORS[int(c) % len(_PYVIS_COLORS)]
                         if isinstance(c, (int, np.integer)) else c)
                     for n, c in zip(nodes, node_color)}

    # Add nodes and edges ------------------------------------------------
    for n in nodes:
        net.add_node(str(n), label=str(n), color=color_map[n])
    for u, v in G.edges():
        net.add_edge(str(u), str(v))

    # Save and display ---------------------------------------------------
    net.save_graph(filename)
    display(HTML(filename))
