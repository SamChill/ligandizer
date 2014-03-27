#!/usr/bin/env python
import ase
from ase.cluster.icosahedron import Icosahedron
from ase.io import write
from ase.optimize import FIRE

from ligandizer import VSEPR, add_ligands

def make_methylamine():
    # Construct Methylamine.
    atoms = ase.Atoms('NH2CH3')

    # Define connectivity.
    atoms_bonds = [ [0,1], [0,2], [0,3], [3,4], [3,5], [3,6] ]
    # Displace atoms so that they aren't on top of each other.
    atoms.rattle(0.001)

    # Construct VSEPR calculator.
    calculator = VSEPR(atoms, atoms_bonds)
    atoms.set_calculator(calculator)
    atoms.center()

    # Run optimization.
    opt = FIRE(atoms)
    opt.run()

    return atoms


ligand = make_methylamine()
nanoparticle = Icosahedron(symbol='Au', noshells=4)
nanoparticle.center(20)

atoms = add_ligands(nanoparticle,
                    ligand,
                    corner_sites=True,
                    edge_sites=[.5],
                    facet_111_sites=[])

write('POSCAR_Ih', atoms)
