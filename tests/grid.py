# from the "Grid single molecule" molgrid tutorial
import torch
import molgrid
from openbabel import pybel as pybel

sdf_file = 'tests/input/benzene.sdf'
mol = next(pybel.readfile('sdf', sdf_file))
c = molgrid.CoordinateSet(mol)
gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(molgrid.defaultGninaLigandTyper.num_types())
grid = torch.zeros(dims, dtype=torch.float32)
gmaker.forward(c.center(), c, grid)
assert grid.norm() > 0
print('OK')

