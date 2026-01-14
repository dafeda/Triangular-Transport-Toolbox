# Triangular Transport Toolbox

<img align="left" src="https://github.com/MaxRamgraber/Triangular-Transport-Toolbox/blob/main/figures/spiral_animated.gif" height="300px">

This repository contains the code for my triangular transport implementation.

## Installation

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MaxRamgraber/Triangular-Transport-Toolbox.git
cd Triangular-Transport-Toolbox
uv sync
```

Then import the class in your Python code:

```python
from triangular_transport_toolbox import transport_map
```

## Examples

The practical use and capabilities of this toolbox are illustrated in example files located in the `examples/` directory:

 - **examples/spiral_distribution/** - illustrates the basic use of the map, from the parameterization of a transport map object to its use for forward mapping, inverse mapping, and conditional sampling.
 - **examples/statistical_inference/** - builds on previously established basics to illustrate the use of transport methods for statistical inference. The first example examines statistical dependencies between temperatures in two cities, the second demonstrates Bayesian parameter inference for Monod kinetics. A third example evaluates the pushforward and pullback densities, the map's approximation to the pdf for the reference and target distributions, respectively.
 - **examples/data_assimilation/** - demonstrates the use of transport maps for Bayesian filtering and smoothing, using the chaotic Lorenz-63 system. These examples also introduce the use of map regularization, the possibility of separation of the map update, and the exploitation of conditional independence.

To run the examples, first install with the examples extras:

```bash
uv sync --extra examples
```

Then run an example:

```bash
uv run python examples/spiral_distribution/Example\ 01\ -\ full\ map/example_01.py
```

---

If you want to learn more about triangular transport maps, I recommend checking out the [**Friendly Introduction to Triangular Transport**](https://arxiv.org/abs/2503.21673), an application-focused tutorial I have authored with colleagues from MIT.

If you're curious about other triangular transport libraries, I recommend checking out the the [**M**onotone **Par**ameterization **T**oolbox MParT](https://measuretransport.github.io/MParT/). It is a joint effort by colleagues from MIT and Dartmouth to create an efficient toolbox for the monotone part of the transport map, realized in C++ for computational efficiency, with bindings to a wide variety of programming languages (Python, MATLAB, Julia). 

---

## Changelog

2025-05-17
Added functions to evaluate the pushforward and pullback densities. Included a new example to showcase these functions in the Examples B folder.