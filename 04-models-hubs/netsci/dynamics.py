"""Dynamics-on-networks helpers for the Network Science Lab Course.

Provides simulation functions for epidemics, information cascades,
immunization strategies, and opinion dynamics on NetworkX graphs.

Functions
---------
sir_ode(y, t, beta, gamma)
    SIR ordinary differential equations (for use with ``odeint``).
network_sir(G, beta, gamma, n_seeds, max_steps, rng)
    Stochastic SIR simulation on a network.
independent_cascade(G, seeds, p, rng)
    Independent cascade (information spreading) model.
immunize_and_simulate(G, fraction, strategy, ...)
    Remove nodes by strategy, then run SIR.
acquaintance_immunize(G, fraction, rng)
    Acquaintance immunization via the friendship paradox.
voter_model(G, max_steps, rng)
    Binary voter model simulation.
"""

import numpy as np
import networkx as nx

from netsci.utils import SEED


def sir_ode(y, t, beta, gamma):
    """SIR ordinary differential equations.

    Parameters
    ----------
    y : array-like
        Current state ``[S, I, R]`` as fractions of the population.
    t : float
        Current time (required by ``odeint``, unused).
    beta : float
        Infection rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    list[float]
        Derivatives ``[dS/dt, dI/dt, dR/dt]``.
    """
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def network_sir(G, beta, gamma, n_seeds=3, max_steps=200, rng=None):
    """Run one stochastic SIR simulation on a network.

    Parameters
    ----------
    G : networkx.Graph
        The contact network.
    beta : float
        Per-edge infection probability per time step.
    gamma : float
        Per-node recovery probability per time step.
    n_seeds : int, default 3
        Number of initially infected nodes.
    max_steps : int, default 200
        Maximum simulation steps.
    rng : numpy.random.Generator or None
        Random number generator.  Uses ``SEED`` if *None*.

    Returns
    -------
    dict
        ``'S'``, ``'I'``, ``'R'`` — lists of counts at each step.
        ``'states'`` — list of dicts mapping node -> ``'S'``/``'I'``/``'R'``.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    nodes = list(G.nodes())
    N = len(nodes)

    # Initialize: pick random seeds
    seeds = set(rng.choice(nodes, size=n_seeds, replace=False))
    S = set(nodes) - seeds
    I = set(seeds)
    R = set()

    S_counts, I_counts, R_counts = [len(S)], [len(I)], [len(R)]
    states_over_time = [{n: ('I' if n in I else 'S') for n in nodes}]

    for step in range(max_steps):
        if len(I) == 0:
            break

        new_I = set()
        new_R = set()

        # Infection: each infected node tries to infect susceptible neighbors
        for node in list(I):
            for neighbor in G.neighbors(node):
                if neighbor in S and rng.random() < beta:
                    new_I.add(neighbor)

        # Recovery: each infected node recovers with probability gamma
        for node in list(I):
            if rng.random() < gamma:
                new_R.add(node)

        S -= new_I
        I = (I | new_I) - new_R
        R = R | new_R

        S_counts.append(len(S))
        I_counts.append(len(I))
        R_counts.append(len(R))
        states_over_time.append({n: ('R' if n in R else 'I' if n in I else 'S') for n in nodes})

    return {"S": S_counts, "I": I_counts, "R": R_counts, "states": states_over_time}


def independent_cascade(G, seeds, p=0.1, rng=None):
    """Run the independent cascade model.

    Each activated node gets exactly one chance to activate each
    neighbor with probability *p*.

    Parameters
    ----------
    G : networkx.Graph
        The network.
    seeds : iterable
        Initially activated nodes.
    p : float, default 0.1
        Activation probability per edge.
    rng : numpy.random.Generator or None
        Random number generator.  Uses ``SEED`` if *None*.

    Returns
    -------
    set
        All activated nodes at the end of the cascade.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    activated = set(seeds)
    newly_activated = set(seeds)
    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated and rng.random() < p:
                    next_activated.add(neighbor)
        activated |= next_activated
        newly_activated = next_activated
    return activated


def immunize_and_simulate(G, fraction, strategy, beta=0.05, gamma=0.1,
                          n_runs=20, n_seeds=3, max_steps=100):
    """Remove nodes by immunization strategy, then simulate SIR.

    Parameters
    ----------
    G : networkx.Graph
        The original network.
    fraction : float
        Fraction of nodes to immunize (remove).
    strategy : str
        ``'random'``, ``'targeted'`` (highest-degree first), or
        anything else for no removal.
    beta, gamma : float
        SIR parameters.
    n_runs : int, default 20
        Number of Monte-Carlo SIR runs.
    n_seeds : int, default 3
        Initial infected nodes per SIR run.
    max_steps : int, default 100
        Maximum SIR steps per run.

    Returns
    -------
    float
        Mean peak infected fraction (relative to original *G* size).
    """
    rng = np.random.default_rng(SEED)
    nodes = list(G.nodes())
    n_remove = int(len(nodes) * fraction)

    if strategy == "random":
        remove = set(rng.choice(nodes, size=n_remove, replace=False))
    elif strategy == "targeted":
        sorted_nodes = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
        remove = set(sorted_nodes[:n_remove])
    else:
        remove = set()

    G_imm = G.copy()
    G_imm.remove_nodes_from(remove)

    if G_imm.number_of_nodes() < n_seeds + 1:
        return 0.0

    peaks = []
    for _ in range(n_runs):
        res = network_sir(G_imm, beta, gamma, n_seeds=n_seeds,
                          max_steps=max_steps, rng=rng)
        peak = max(res["I"]) / G.number_of_nodes()  # fraction of ORIGINAL network
        peaks.append(peak)
    return np.mean(peaks)


def acquaintance_immunize(G, fraction, rng):
    """Acquaintance immunization via the friendship paradox.

    Pick a random node, vaccinate one of its random neighbors.  Repeat
    until the desired fraction of nodes is immunized.

    Parameters
    ----------
    G : networkx.Graph
        The network.
    fraction : float
        Target fraction of nodes to immunize.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    networkx.Graph
        A copy of *G* with immunized nodes removed.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    n_target = int(N * fraction)
    immunized = set()

    while len(immunized) < n_target:
        # Pick random node
        person = rng.choice(nodes)
        neighbors = list(G.neighbors(person))
        if neighbors:
            # Vaccinate a random neighbor
            friend = rng.choice(neighbors)
            immunized.add(friend)

    G_imm = G.copy()
    G_imm.remove_nodes_from(immunized)
    return G_imm


def voter_model(G, max_steps=50000, rng=None):
    """Run the binary voter model on graph *G*.

    Each node starts with a random opinion (0 or 1).  At each step a
    random node copies a random neighbor's opinion.

    Parameters
    ----------
    G : networkx.Graph
        The network.
    max_steps : int, default 50000
        Maximum number of update steps.
    rng : numpy.random.Generator or None
        Random number generator.  Uses ``SEED`` if *None*.

    Returns
    -------
    dict
        ``'fraction_1'`` — fraction of nodes with opinion 1 (sampled
        every 50 steps).
        ``'states'`` — opinion dict at each sampled step.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    nodes = list(G.nodes())
    N = len(nodes)

    # Initialize: half opinion 0, half opinion 1 (random split)
    opinions = {n: int(rng.random() < 0.5) for n in nodes}
    fraction_1 = [sum(opinions.values()) / N]
    states = [dict(opinions)]

    for step in range(1, max_steps + 1):
        # Pick a random node
        node = rng.choice(nodes)
        neighbors = list(G.neighbors(node))
        if neighbors:
            # Copy a random neighbor's opinion
            neighbor = rng.choice(neighbors)
            opinions[node] = opinions[neighbor]

        # Record every 50 steps to keep output manageable
        if step % 50 == 0:
            frac = sum(opinions.values()) / N
            fraction_1.append(frac)
            states.append(dict(opinions))
            if frac == 0.0 or frac == 1.0:
                break  # consensus reached

    return {"fraction_1": fraction_1, "states": states}
