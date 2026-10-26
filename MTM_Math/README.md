# MTM_Math

Numerical and symbolic calculations for the mesoscopic tensorial model:
energy density, stress, forces, tangent operators and matrix reduction.
The package is included in MTSPlot and can be used independently of plotting or
simulation management.

Install MTSPlot with its `math` extra to include NumPy and SymPy. From the
MTSPlot repository directory:

```sh
python -m pip install -e ../MTSManager -e '.[math]'
```

## Energy and deformation

```python
from MTM_Math.energy import ContiEnergy

energy = ContiEnergy.energy_from_simple_shear(0.1)
```

Energy functions return densities. MTS2D's macro `avg_energy` is mean energy per
triangle; account for reference area when comparing the two. Physical
fixed-coordinate stress requires the full total deformation matrix, including
its rotation.

## Metric reduction

```python
import numpy as np
from MTM_Math.reduction import lagrange_reduction, plastic_reduction

F = np.array([[1.0, 2.25], [0.0, 1.0]])
C = F.T @ F
C_reduced, M = plastic_reduction(C, compute_M=True)
```

Plastic reduction applies unit integer shears until the metric satisfies
`abs(C12) <= min(C11, C22) / 2`, choosing a horizontal shear when the diagonal
entries tie. With `compute_M=True`, the returned transform satisfies
`C_reduced = M.T @ C @ M`. With the default `compute_M=False`, the second return
value is a small zero placeholder, not a transformation matrix.

Lagrange reduction additionally maps the metric to a fundamental domain with
nonnegative off-diagonal entry and ordered diagonals. Both routines accept arrays
of symmetric 2×2 metrics and leave inputs unchanged. Component-level functions
are also available. Invalid metrics and failure to converge raise errors.

Mathematical checks and comparisons with the C++ simulator are in MTSPlot/tests.
