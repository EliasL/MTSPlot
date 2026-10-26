# MTM_Math

Energy, stress, tangent and Lagrange reduction for the mesoscopic tensorial model.
These modules belong to the MTSPlot repository and distribution but do not import
plotting or simulation-management code. Install `MTSPlot[math]` alongside the
manager using the workspace setup instructions.

```python
from MTM_Math.energy import ContiEnergy, PieceWiseQuadratic
from MTM_Math.reduction import lagrange_reduction
```

Energy functions return densities. MTS2D macro avg_energy is mean energy per
triangle; account for reference area when comparing them. Physical fixed-coordinate
stress requires the full total deformation matrix T, including its rotation.

The mathematical and simulator-comparison tests live in MTSPlot/tests.
