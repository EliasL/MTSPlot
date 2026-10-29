"""Energy comparisons with the current simulator, owned by MTM_Math."""

from pathlib import Path
import subprocess
import tempfile
import unittest
import os
import numpy as np

from MTS_Management.config import Config
from MTS_Management.data import read_macros
from MTS_Management.provenance import verify_schema
from MTS_Management.runner import PACKAGE_ROOT
from MTM_Math.energy import ContiEnergy

SOURCE = PACKAGE_ROOT.parent / "MTS2D"
EXE = Path(os.environ.get("MTS_TEST_BINARY", SOURCE / "build/MTS2D")).resolve()


class CurrentEnergyTests(unittest.TestCase):
    @unittest.skipUnless(EXE.exists(),
                         "Build the current simulator for energy integration tests")
    def test_macro_energy_density(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config(rows=4, cols=4, initialGuessNoise=0, boundary="periodic",
                            totalLoad=.003, loadIncrement=.001, loading="noRelaxation",
                            writeDumps=False, writeMeshVTUs=False)
            folder = root / config.folder_name()
            folder.mkdir()
            config.write(folder / "config.conf")
            subprocess.run([str(EXE), str(folder)], check=True, capture_output=True, text=True)
            rows = read_macros(folder / "macroData.csv")
            for row in rows:
                gamma = float(row["load"])
                energy = ContiEnergy.energy_from_simple_shear(gamma)
                # MTS2D stores mean energy per triangle; this mesh has area 1/2.
                self.assertAlmostEqual(float(row["avg_energy"]), .5*energy, places=9)

    @unittest.skipUnless(SOURCE.exists(), "Current simulator source is not beside this checkout")
    def test_cpp_energy_formula_and_schema(self):
        verify_schema(SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"reference.cpp"
            source.write_text('''#include "Simulation/energyFunctions.h"
#include <iostream>
#include <iomanip>
int main() {
  std::cout << std::setprecision(17);
  for (double c12 : {0.0, 0.1, 0.3}) {
    std::cout << ContiPotential::energyDensityFormula(1.1, 1.03, c12, -0.25, 4.0, 1.2) << "\\n";
  }
}
''')
            subprocess.run(["c++", "-std=c++17", "-I", str(SOURCE/"src"),
                            "-I", str(SOURCE/"libs/eigen"), str(source), "-o", str(root/"reference")], check=True)
            actual = np.fromstring(subprocess.check_output([str(root/"reference")], text=True), sep="\n")
            expected = [ContiEnergy.energy_from_reduced_C_components(1.1, 1.03, c12,
                        noise=1.2, zeroReference=False) for c12 in (0, .1, .3)]
            np.testing.assert_allclose(actual, expected, rtol=1e-14)
