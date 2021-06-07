import sys, os, pytest
from numpy import isclose, allclose
import torch

sys.path.insert(0, '.')
from liGAN.atom_types import AtomTyper
from liGAN.atom_structs import AtomStruct


class TestAtomStruct(object):

    @pytest.fixture
    def typer(self):
        typer = AtomTyper.get_typer(prop_funcs='', radius_func=1.0)
        typer.elem_range = [8]
        return typer

    @pytest.fixture
    def struct(self, typer):
        return AtomStruct(
            coords=torch.zeros((typer.n_types, 3)),
            types=torch.eye(typer.n_types),
            typer=typer
        )

    def test_init(self, struct):
        assert struct.atom_types == struct.atom_types
        assert struct.atom_types is struct.atom_types

        assert (struct.atomic_radii == struct.atomic_radii).all()
        assert struct.atomic_radii is struct.atomic_radii
