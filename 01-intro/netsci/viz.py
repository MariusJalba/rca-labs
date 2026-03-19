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
draw_graph_anatomy(G, seed=SEED)
    Annotated tutorial diagram labeling node, edge, and degree.
draw_weighted_graph(G, weight_attr="weight", title=None, seed=SEED, ...)
    Node-link diagram with edge width/color proportional to weight.
plot_adjacency(G, title=None, cmap="Blues", nodelist=None, group_labels=None, ...)
    Adjacency matrix with optional node reordering and group annotations.
plot_in_out_degree(G, title_prefix="", ...)
    Side-by-side in-degree / out-degree histograms for directed graphs.
compare_layouts(G, node_color=None, title=None)
    Three-panel comparison of spring, Kamada-Kawai, and circular layouts.
draw_bipartite(G, top_nodes, bottom_nodes, title=None, ...)
    Two-column bipartite layout with typed node shapes and legend.
draw_projection(G, bipartite_G=None, title=None)
    One-mode projection plot that highlights isolates in red.
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
        fig, ax = plt.subplots(figsize=(7, 4.5))

        if log:
            deg_count = Counter(degrees)
            x, y = zip(*sorted(deg_count.items()))
            ax.scatter(
                x,
                y,
                s=40,
                alpha=0.8,
                color="#4878CF",
                edgecolors="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                title or "Degree Distribution (log-log)",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
        else:
            ax.hist(
                degrees,
                bins=range(min(degrees), max(degrees) + 2),
                edgecolor="white",
                alpha=0.85,
                color="#4878CF",
                linewidth=0.8,
            )
            ax.set_title(
                title or "Degree Distribution", fontsize=13, fontweight="bold", pad=10
            )

        ax.set_xlabel("Degree (k)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
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
        ax.scatter(ccdf_x, ccdf_y, s=20, alpha=0.7, edgecolors="white", linewidth=0.5)

        if fit_line:
            k_min = min(degrees)
            log_ratios = np.log(np.array(degrees) / k_min)
            alpha = 1 + n / log_ratios.sum()
            k_range = np.logspace(np.log10(k_min), np.log10(max(degrees)), 200)
            fit_y = (k_range / k_min) ** (-(alpha - 1))
            ax.plot(k_range, fit_y, "r--", lw=1.5, label=f"MLE fit  α = {alpha:.2f}")
            ax.legend(fontsize=9, framealpha=0.9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)", fontsize=11)
        ax.set_ylabel("P(K ≥ k)", fontsize=11)
        ax.set_title(
            title or "Degree CCDF (log-log)", fontsize=13, fontweight="bold", pad=10
        )
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Node-link graph drawing
# ---------------------------------------------------------------------------


def draw_graph(G, node_color=None, node_size=None, title=None, layout="spring"):
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

    n_nodes = G.number_of_nodes()
    _node_size = node_size if node_size is not None else max(300 - n_nodes * 2, 40)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw edges — with arrows for directed graphs
        edge_kw = dict(
            ax=ax,
            edge_color="#aaaaaa",
            width=0.7,
            alpha=0.5,
        )
        if G.is_directed():
            edge_kw.update(
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                connectionstyle="arc3,rad=0.1",
                min_source_margin=8,
                min_target_margin=8,
            )
        nx.draw_networkx_edges(G, pos, **edge_kw)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_color or "#4878CF",
            node_size=_node_size,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.92,
        )
        if n_nodes <= 50:
            nx.draw_networkx_labels(
                G,
                pos,
                ax=ax,
                font_size=max(10 - n_nodes // 10, 7),
                font_color="#2c3e50",
            )
        elif n_nodes <= 200:
            # Show labels for top-degree nodes only
            degrees = dict(G.degree())
            top_k = min(10, n_nodes // 5)
            top = sorted(degrees, key=degrees.get, reverse=True)[:top_k]
            labels = {n: str(n) for n in top}
            nx.draw_networkx_labels(
                G,
                pos,
                labels,
                ax=ax,
                font_size=9,
                font_weight="bold",
                font_color="#2c3e50",
            )
        ax.set_title(title or "", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Adjacency matrix heatmap
# ---------------------------------------------------------------------------


def plot_adjacency(
    G, title=None, cmap="Blues", nodelist=None, group_labels=None, group_colors=None
):
    """Plot a heatmap of the adjacency matrix of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    cmap : str, default "Blues"
        Matplotlib/seaborn colormap name.  Ignored when *group_labels* is
        given (each block gets its own colour instead).
    nodelist : list, optional
        Order of nodes in the matrix rows/columns.  Sorting by community
        membership reveals block-diagonal structure.
    group_labels : list of (str, int), optional
        List of ``(label, count)`` tuples describing consecutive groups in
        *nodelist*.  E.g. ``[("Mr. Hi", 17), ("Officer", 17)]``.
        When provided the **heatmap cells themselves** are coloured by group:
        each on-diagonal block gets its own colour and cross-group edges are
        shown in grey, making the community structure immediately obvious.
    group_colors : list of str, optional
        One colour per group.  Defaults to ``["#D65F5F", "#4878CF", ...]``.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    A = nx.to_numpy_array(G, nodelist=nodelist)
    A = (A > 0).astype(float)  # binary
    n = A.shape[0]

    default_colors = ["#D65F5F", "#4878CF", "#6ACC65", "#B47CC7", "#C4AD66"]

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(7.5, 7))

        if group_labels is not None:
            colors = group_colors or default_colors

            # Build an RGBA image: background, cross-group edges, per-block edges
            bg = np.array([0.96, 0.96, 0.96, 1.0])  # light grey background
            cross_c = np.array([0.72, 0.72, 0.72, 1.0])  # grey for cross-group
            img = np.tile(bg, (n, n, 1))

            # Assign group index to each node
            group_idx = np.zeros(n, dtype=int)
            row = 0
            for gi, (_, count) in enumerate(group_labels):
                group_idx[row : row + count] = gi
                row += count

            for i in range(n):
                for j in range(n):
                    if A[i, j] > 0:
                        gi, gj = group_idx[i], group_idx[j]
                        if gi == gj:
                            # Intra-group edge → group colour
                            rgb = to_rgb(colors[gi % len(colors)])
                            img[i, j] = (*rgb, 1.0)
                        else:
                            # Cross-group edge → grey
                            img[i, j] = cross_c

            ax.imshow(img, interpolation="nearest", aspect="equal", origin="upper")

            # Draw block boundaries and bracket labels
            row = 0
            for gi, (label, count) in enumerate(group_labels):
                c = colors[gi % len(colors)]
                mid = row + count / 2 - 0.5

                # Label on the left
                ax.text(
                    -1.5,
                    mid,
                    label,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=c,
                    clip_on=False,
                )

                # Separator line after each group (except last)
                if gi < len(group_labels) - 1:
                    sep = row + count - 0.5
                    ax.axhline(sep, color="white", linewidth=2)
                    ax.axvline(sep, color="white", linewidth=2)

                row += count

            # Legend
            handles = [
                Patch(facecolor=colors[i % len(colors)], label=lbl)
                for i, (lbl, _) in enumerate(group_labels)
            ]
            handles.append(Patch(facecolor="#B8B8B8", label="Cross-group"))
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
                edgecolor="#cccccc",
            )

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Plain heatmap (no groups)
            sns.heatmap(
                A,
                ax=ax,
                cmap=cmap,
                square=True,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
            )
            ax.set_ylabel("Node", fontsize=11)

        ax.set_xlabel("Node", fontsize=11)
        ax.set_title(
            title or "Adjacency Matrix", fontsize=13, fontweight="bold", pad=10
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Annotated graph anatomy (tutorial helper)
# ---------------------------------------------------------------------------


def draw_graph_anatomy(G, seed=SEED):
    """Draw an annotated tutorial diagram labeling node, edge, and degree.

    Designed for a small (≤10 node) teaching graph with named nodes
    including 'Alice', 'Bob', and 'Carol'.
    """
    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=seed)

        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=600,
            edgecolors="white",
            linewidths=2.0,
            alpha=0.95,
        )

        label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            ax=ax,
            font_size=10,
            font_color="#2c3e50",
            font_weight="bold",
        )

        # Annotate "node" → first node, "edge" → midpoint, "degree" → second node
        nodes = list(G.nodes())
        n0, n1, n2 = nodes[0], nodes[1], nodes[2]

        ax.annotate(
            "node",
            xy=pos[n0],
            xytext=(pos[n0][0] - 0.35, pos[n0][1] + 0.30),
            fontsize=13,
            fontweight="bold",
            color="#e74c3c",
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2),
        )

        mid = ((pos[n1][0] + pos[n2][0]) / 2, (pos[n1][1] + pos[n2][1]) / 2)
        ax.annotate(
            "edge",
            xy=mid,
            xytext=(mid[0] + 0.35, mid[1] + 0.25),
            fontsize=13,
            fontweight="bold",
            color="#27ae60",
            arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2),
        )

        ax.annotate(
            f"degree = {G.degree(n1)}",
            xy=pos[n1],
            xytext=(pos[n1][0] + 0.38, pos[n1][1] - 0.28),
            fontsize=13,
            fontweight="bold",
            color="#8e44ad",
            arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2),
        )

        ax.set_title(
            "Graph Anatomy: Nodes, Edges, and Degree",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        ax.margins(0.15)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Weighted graph drawing
# ---------------------------------------------------------------------------


def draw_weighted_graph(
    G,
    weight_attr="weight",
    title=None,
    seed=SEED,
    top_labels=8,
    top_weight_edges=5,
    k=0.3,
):
    """Draw a node-link diagram with edge width/color proportional to weight.

    Labels are shown for top-degree nodes *and* nodes involved in the
    heaviest edges, so important connections are always visible.

    Parameters
    ----------
    G : networkx.Graph
    weight_attr : str, default "weight"
    title : str, optional
    seed : int
    top_labels : int
        Number of top-degree nodes to label.
    top_weight_edges : int
        Number of heaviest edges whose endpoints are also labeled.
    k : float
        Spring layout spacing parameter.
    """
    weights = [G[u][v][weight_attr] for u, v in G.edges()]
    max_w = max(weights)
    norm_widths = [0.3 + 4.0 * w / max_w for w in weights]
    edge_colors = [plt.cm.YlOrRd(0.2 + 0.7 * w / max_w) for w in weights]

    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 25 for n in G.nodes()]

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(10, 7.5))
        pos = nx.spring_layout(G, seed=seed, k=k)

        nx.draw_networkx_edges(
            G, pos, ax=ax, width=norm_widths, edge_color=edge_colors, alpha=0.75
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color="#4878CF",
            alpha=0.90,
            edgecolors="white",
            linewidths=0.8,
        )

        # Label top-degree + top-weight-edge nodes
        top_by_degree = set(sorted(degrees, key=degrees.get, reverse=True)[:top_labels])
        top_edges = sorted(G.edges(data=weight_attr), key=lambda e: e[2], reverse=True)[
            :top_weight_edges
        ]
        top_by_weight = {u for u, v, w in top_edges} | {v for u, v, w in top_edges}
        label_nodes = top_by_degree | top_by_weight

        label_pos = {n: (pos[n][0], pos[n][1] + 0.04) for n in label_nodes}
        nx.draw_networkx_labels(
            G,
            label_pos,
            {n: n for n in label_nodes},
            ax=ax,
            font_size=9,
            font_weight="bold",
            font_color="#2c3e50",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )

        ax.set_title(title or "Weighted Graph", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# In-degree / Out-degree comparison (directed graphs)
# ---------------------------------------------------------------------------


def plot_in_out_degree(G, title_prefix=""):
    """Plot side-by-side in-degree and out-degree histograms.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph.
    title_prefix : str
        Prefix for subplot titles (e.g. ``"Email"``).
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    prefix = f"{title_prefix} — " if title_prefix else ""

    with plt.style.context("seaborn-v0_8-muted"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        axes[0].hist(
            in_degrees,
            bins=50,
            edgecolor="white",
            alpha=0.85,
            color="#4878CF",
            linewidth=0.6,
        )
        axes[0].set_xlabel("In-degree", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title(
            f"{prefix}In-degree Distribution\n(How many emails received)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        axes[1].hist(
            out_degrees,
            bins=50,
            edgecolor="white",
            alpha=0.85,
            color="#D65F5F",
            linewidth=0.6,
        )
        axes[1].set_xlabel("Out-degree", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title(
            f"{prefix}Out-degree Distribution\n(How many emails sent)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Layout comparison (3-panel)
# ---------------------------------------------------------------------------


def compare_layouts(G, node_color=None, title=None):
    """Draw the same graph with spring, Kamada-Kawai, and circular layouts.

    Parameters
    ----------
    G : networkx.Graph
    node_color : color or list of colors, optional
    title : str, optional
        Overall suptitle.
    """
    n = G.number_of_nodes()
    _size = max(180, 300 - n * 3)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        layouts = {
            "Spring": nx.spring_layout(G, seed=SEED),
            "Kamada-Kawai": nx.kamada_kawai_layout(G),
            "Circular": nx.circular_layout(G),
        }

        for ax, (name, pos) in zip(axes, layouts.items()):
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color="#aaaaaa", width=0.7, alpha=0.4
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_color or "#4878CF",
                node_size=_size,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.92,
            )
            if n <= 50:
                nx.draw_networkx_labels(
                    G, pos, ax=ax, font_size=7, font_color="#2c3e50"
                )
            ax.set_title(name, fontsize=14, fontweight="bold", pad=8)
            ax.axis("off")

        fig.suptitle(
            title or "Same Graph, Three Layouts", fontsize=16, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def draw_bipartite(
    G, top_nodes, bottom_nodes, title=None, top_label="Type A", bottom_label="Type B"
):
    """Draw a two-column bipartite layout.

    Parameters
    ----------
    G : networkx.Graph
        A bipartite graph.
    top_nodes : list
        Nodes to place in the left column (drawn as squares).
    bottom_nodes : list
        Nodes to place in the right column (drawn as circles).
    title : str, optional
        Plot title.
    top_label, bottom_label : str
        Legend labels for the two node types.
    """
    from matplotlib.lines import Line2D

    pos = {}
    for i, n in enumerate(top_nodes):
        pos[n] = (0, -i * 1.2)
    for i, n in enumerate(bottom_nodes):
        pos[n] = (2.0, -i * 0.9)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(9, 5))

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=top_nodes,
            ax=ax,
            node_color="#4878CF",
            node_shape="s",
            node_size=600,
            edgecolors="white",
            linewidths=2.0,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bottom_nodes,
            ax=ax,
            node_color="#D65F5F",
            node_size=500,
            edgecolors="white",
            linewidths=2.0,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=10,
            font_weight="bold",
            font_color="#2c3e50",
        )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#4878CF",
                markersize=12,
                markeredgecolor="white",
                label=top_label,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#D65F5F",
                markersize=12,
                markeredgecolor="white",
                label=bottom_label,
            ),
        ]
        ax.legend(
            handles=legend_elements,
            fontsize=11,
            loc="upper right",
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        ax.set_title(title or "", fontsize=14, fontweight="bold", pad=14)
        ax.margins(0.15)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


def draw_projection(G, bipartite_G=None, title=None):
    """Draw a one-mode projection, highlighting isolates in red.

    Parameters
    ----------
    G : networkx.Graph
        The projected (one-mode) graph.
    bipartite_G : networkx.Graph, optional
        The original bipartite graph.  If provided, shared neighbors are
        printed for each edge.
    title : str, optional
        Plot title.
    """
    # Print edge info
    if bipartite_G is not None:
        for u, v in G.edges():
            shared = set(bipartite_G.neighbors(u)) & set(bipartite_G.neighbors(v))
            print(f"  {u} -- {v}  (shared: {shared})")

    isolates = list(nx.isolates(G))
    if isolates:
        print(f"\nIsolated nodes (no connections): {isolates}")

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G, seed=SEED, k=1.5)

        # Move isolates near the cluster center so they're visible
        if isolates:
            center = np.mean(list(pos.values()), axis=0)
            for iso in isolates:
                pos[iso] = center + np.array([0.4, -0.3])

        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=500,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.92,
        )
        if isolates:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=isolates,
                ax=ax,
                node_color="#D65F5F",
                node_size=500,
                edgecolors="white",
                linewidths=1.5,
                alpha=0.92,
            )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=10,
            font_color="#2c3e50",
            font_weight="bold",
        )

        ax.set_title(title or "", fontsize=13, fontweight="bold", pad=10)
        ax.margins(0.2)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Interactive PyVis visualization
# ---------------------------------------------------------------------------

# Default color palette for community coloring (10 distinct colors).
_PYVIS_COLORS = [
    "#4878CF",
    "#6ACC65",
    "#D65F5F",
    "#B47CC7",
    "#C4AD66",
    "#77BEDB",
    "#E8A06B",
    "#8C8C8C",
    "#C572D1",
    "#64B5CD",
]


def draw_pyvis(G, node_color=None, title=None, height="500px", filename="graph.html"):
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
    from IPython.display import HTML, display
    from pyvis.network import Network

    net = Network(height=height, width="100%", notebook=True, cdn_resources="in_line")
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
        color_map = {
            n: (
                _PYVIS_COLORS[int(c) % len(_PYVIS_COLORS)]
                if isinstance(c, (int, np.integer))
                else c
            )
            for n, c in zip(nodes, node_color)
        }

    # Add nodes and edges ------------------------------------------------
    for n in nodes:
        net.add_node(str(n), label=str(n), color=color_map[n])
    for u, v in G.edges():
        net.add_edge(str(u), str(v))

    # Save and display ---------------------------------------------------
    net.save_graph(filename)
    display(HTML(filename))
