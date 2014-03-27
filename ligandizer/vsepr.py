#!/usr/bin/env python
from collections import namedtuple
from ase.data import covalent_radii, atomic_numbers, vdw_radii
from ase import Atoms
#from tsase.optimize import SDLBFGS
from ase.optimize import FIRE
import numpy
numpy.seterr(all='raise')
import os

class VSEPR:

    def __init__(self, atoms, bonds, angles='auto', nonbonded='auto'):
        if angles == 'auto':
            angles = connect_angles(bonds)
        if nonbonded == 'auto':
            nonbonded = connect_nonbonded(atoms, angles)

        atom_types = determine_atom_types(atoms, bonds)

        k_bonds = []
        r_bonds = []
        for bond in bonds:
            i,j=bond
            r_bond,k_bond = bond_stretch_parameter(atom_types[i], atom_types[j])
            r_bonds.append(r_bond)
            k_bonds.append(k_bond)
        self.k_bonds = numpy.array(k_bonds)
        self.r_bonds = numpy.array(r_bonds)

        k_angles = []
        theta_angles = []
        for angle in angles:
            i,j,k=angle
            theta_angle, k_angle = angle_bending_parameter(atom_types[i],
                                                           atom_types[j],
                                                           atom_types[k])
            k_angles.append(k_angle)
            theta_angles.append(theta_angle)
        self.k_angles = numpy.array(k_angles)
        self.theta_angles = numpy.array(theta_angles)

        sigmas = []
        r_vdws = []
        for nonbond in nonbonded:
            i,j=nonbond
            r_vdw, sigma = nonbonded_parameter(atom_types[i], atom_types[j])
            sigmas.append(sigma)
            r_vdws.append(r_vdw)
        self.sigmas = numpy.array(sigmas)
        self.r_vdws = numpy.array(r_vdws)

        self.nbonds = len(bonds)
        self.nangles = len(angles)
        self.nnonbonded = len(nonbonded)

        self.bonds = numpy.array(bonds,numpy.int32).ravel()
        self.angles = numpy.array(angles,numpy.int32).ravel()
        self.nonbonded = numpy.array(nonbonded,numpy.int32).ravel()

        self.force_calls = 0
        self.atoms = None

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if self.calculation_required(atoms, "energy"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.u

    def get_forces(self, atoms):
        if self.calculation_required(atoms, "forces"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.f.copy()

    def get_stress(self, atoms):
        raise NotImplementedError

    def calculation_required(self, atoms, quantities):
        if atoms != self.atoms or self.atoms == None:
            return True
        if self.f == None or self.u == None or atoms == None:
            return True
        return False

    def set_atoms(self, atoms):
        pass

    def calculate(self):
        #vsepr(double *positions, double *forces, int num_atoms, double *energy,
        #     int *bonds_list, int num_bonds, double *k_bonds, double *r_bonds,
        #     int *angles_list, int num_angles, double *k_angles, double *theta_angles,
        #     int *nonbonded_list, int num_nonbonds, double *sigmas, double *r_vdws)
        import ctypes
        from numpy.ctypeslib import ndpointer
        u = numpy.empty(1)
        f = numpy.zeros((len(self.atoms), 3))

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_vsepr.so')
        lib = ctypes.cdll.LoadLibrary(path)
        vsepr_ = lib.vsepr
        vsepr_.restype = None
        vsepr_.argtypes = [ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_double),
                           ctypes.c_int,
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_int),
                           ctypes.c_int,
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_int),
                           ctypes.c_int,
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_int),
                           ctypes.c_int,
                           ndpointer(ctypes.c_double),
                           ndpointer(ctypes.c_double),]

        positions = self.atoms.positions.ravel()
        vsepr_(positions, f, len(self.atoms), u,
               self.bonds, self.nbonds, self.k_bonds, self.r_bonds,
               self.angles, self.nangles, self.k_angles, self.theta_angles,
               self.nonbonded, self.nnonbonded, self.sigmas, self.r_vdws)

        self.u = u[0]
        self.f = f
        self.force_calls += 1


def bond_stretch_parameter(*atom_types):
    key = ''.join(sorted([atom_types[0].symbol, atom_types[1].symbol]))
    # Bond stretch force constants in eV/Ang^2.
    k_ij = {
        'X'  : 25.0,
        'CH' : 32.0,
        'CC' : 23.0,
        'HO' : 30.0,
    }

    r_ij = {
        'AuN' : 2.336,
        'CN'  : 1.491,
        'HN'  : 1.026,
        'CH'  : 1.109,
    }

    if key in k_ij:
        k_r = k_ij[key]
    else:
        k_r = k_ij['X']

    if key in r_ij:
        r_eq = r_ij[key]
    else:
        za = atomic_numbers[atom_types[0].symbol]
        zb = atomic_numbers[atom_types[1].symbol]
        r_eq = covalent_radii[za] + covalent_radii[zb]

    return r_eq, k_r

def nonbonded_parameter(*atom_types):
    zi = atomic_numbers[atom_types[0].symbol]
    zj = atomic_numbers[atom_types[1].symbol]

    if zi == 'N' and zj == 'N':
        return 4.5, 1e-3

    r_vdw = 1.1*(vdw_radii[zi] + vdw_radii[zj])

    C6 = {
        'H'  : 0.00086728,
        'C'  : 0.0052037,
        'O'  : 0.0086728,
        'N'  : 0.0069383,
        'Au' : 0.007,
        'X'  : 0.007,
    }

    if atom_types[0].symbol in C6:
        C6_i = C6[atom_types[0].symbol]
    else:
        C6_i = C6['X']

    if atom_types[1].symbol in C6:
        C6_j = C6[atom_types[1].symbol]
    else:
        C6_j = C6['X']

    C6_ij = numpy.sqrt(C6_i * C6_j)

    return r_vdw, C6_ij


def angle_bending_parameter(*atom_types):
    hybridization = atom_types[1].hybridization
    theta_eq = {  'sp' : 180.0 * numpy.pi/180.0,
                 'sp2' : 120.0 * numpy.pi/180.0,
                 'sp3' : 109.5 * numpy.pi/180.0 }[hybridization]

    key = ''.join([ t.symbol for t in atom_types ])

    # Bond bending force constants in eV/radian^2.
    k_ijk = {
        'XXX' : 100e-5*(180.0/numpy.pi)**2,
        'HCH' : 147e-5*(180.0/numpy.pi)**2,
        'CCH' :  98e-5*(180.0/numpy.pi)**2,
        'HOH' : 100e-5*(180.0/numpy.pi)**2,
        'CCC' :  85e-5*(180.0/numpy.pi)**2,
    }

    if key in k_ijk:
        k_theta = k_ijk[key]
    else:
        k_theta = k_ijk['XXX']

    return theta_eq, k_theta

AtomType = namedtuple('AtomType', ['symbol', 'hybridization'])
def determine_atom_types(atoms, bonds):

    valence_electrons = numpy.zeros(len(atoms), int)
    for i in xrange(len(atoms)):
        z = atoms[i].number
        if z <= 2:
            valence_electrons[i] = z
        elif z <= 10:
            valence_electrons[i] = z - 2
        elif z <= 18:
            valence_electrons[i] = z - 10

    bond_orders = numpy.zeros(len(atoms), int)
    coordination_numbers = numpy.zeros(len(atoms), int)
    # Account for single bonds.
    for bond in bonds:
        bond_orders[bond] += 1
        coordination_numbers[bond] += 1
        i,j = bond

        # If we bond to a metal.
        if valence_electrons[i] == 0:
            valence_electrons[j] -= 1
        elif valence_electrons[j] == 0:
            valence_electrons[i] -= 1

    def satisifies_octet_rule(i):
        if atoms[i].number <= 2:
            shell = 2
        else:
            shell = 8

        if valence_electrons[i] + bond_orders[i] < shell:
            return False
        else:
            return True

    # Account for double bonds.
    for bond in bonds:
        if satisifies_octet_rule(bond[0]): continue
        if satisifies_octet_rule(bond[1]): continue
        bond_orders[bond] += 1

    # Account for triple bonds.
    for bond in bonds:
        if satisifies_octet_rule(bond[0]): continue
        if satisifies_octet_rule(bond[1]): continue
        bond_orders[bond] += 1

    atom_types = []
    for i in xrange(len(atoms)):
        if valence_electrons[i] == 0:
            atom_types.append(AtomType(atoms[i].symbol, 'metal'))
            continue

        bond_order = bond_orders[i]
        if bond_order == 0 and atoms[i].number < 20:
            hybrid = 'undefined'
            atom_types.append(AtomType(atoms[i].symbol, 'undefined'))
            continue

        unpaired_electrons = valence_electrons[i] - bond_order
        lone_pairs = unpaired_electrons/2.0


        try:
            hybrid = { 1:'s',
                       2:'sp',
                       3:'sp2',
                       4:'sp3' }[coordination_numbers[i]+lone_pairs]
        except KeyError:
            hybrid = 'sp3'
        atom_type = AtomType(atoms[i].symbol, hybrid)
        atom_types.append(atom_type)

    return atom_types

def cutoff_distance(atom1, atom2, fudge_factor=1.2):
    z1 = atom1.number
    z2 = atom2.number
    r_cut = fudge_factor*(covalent_radii[z1] + covalent_radii[z2])
    return r_cut

def connect_bonds(atoms, indices=None, mask=None):
    bonds = []

    if indices is None:
        indices = xrange(len(atoms))

    if mask is not None:
        assert(len(atoms) == len(mask))
        indices = [ i for i in xrange(len(mask)) if mask[i] == True ]

    for ii in xrange(len(indices)):
        for jj in xrange(ii+1, len(indices)):
            i = indices[ii]
            j = indices[jj]
            r_cut = cutoff_distance(atoms[i], atoms[j])
            r = atoms.get_distance(i, j, mic=True)
            if r < r_cut:
                bonds.append([i,j])

    return bonds

def connect_angles(bonds):
    angles = []

    for i in xrange(len(bonds)):
        for j in xrange(i+1, len(bonds)):
            if not (bonds[i][0] in bonds[j] or bonds[i][1] in bonds[j]):
                continue

            bridge = set(bonds[i]) & set(bonds[j])
            angle = [ list(set(bonds[i]) - bridge)[0],
                      list(bridge)[0],
                      list(set(bonds[j]) - bridge)[0] ]

            angles.append(angle)


    return angles

def connect_nonbonded(atoms, angles, r_cut=None, mask=None):
    nonbonded = []
    for i in xrange(len(atoms)):
        if not mask is None and mask[i] == False: continue
        for j in xrange(i+1, len(atoms)):
            skip = False
            for angle in angles:
                if i in angle and j in angle:
                    skip = True
                    break
            if skip: continue
            if r_cut is not None:
                r = atoms.get_distance(i,j,mic=True)
                if r < r_cut:
                    nonbonded.append([i,j])
            else:
                nonbonded.append([i,j])
    return nonbonded

def opt(atoms, bonds):
    atoms.rattle()
    calculator = VSEPR(atoms, bonds)
    atoms.set_calculator(calculator)
    opt = FIRE(atoms,  dtmax=0.5, logfile='/dev/null', trajectory='opt.traj')
    opt.run(fmax=0.1)

def make_pamam(generations):
    # Make core.
    atoms = Atoms('NCHHCHHN')

    # Define connectivity of core.
    bonds = [ [0,1], [1,2], [1,3], [1,4], [4,5], [4,6], [4,7] ]

    terminal_atoms = [0, 7]
    opt(atoms, bonds)

    monomer_positions = None
    for generation in xrange(generations+1):
        new_terminal_atoms = []
        for terminal_atom in terminal_atoms:
            for branch in xrange(2):
                print 'generation %i terminal atom: %3i branch: %3i' %(generation, terminal_atom, branch)
                # Append a monomer branch.
                monomer = Atoms('CHHCHHCONHCHHCHHN')
                monomer_bonds = [ [0,1], [0,2], [0,3],
                                  [3,4], [3,5], [3,6],
                                  [6,7], [6,8],
                                  [8,9], [8,10],
                                  [10,11], [10,12], [10,13],
                                  [13,14], [13,15], [13,16] ]
                if monomer_positions is None:
                    opt(monomer, monomer_bonds)
                    monomer_positions = monomer.get_positions()
                else:
                    monomer.set_positions(monomer_positions)
                for bond in monomer_bonds:
                    bond[0] += len(atoms)
                    bond[1] += len(atoms)

                bonds += [[terminal_atom, len(atoms)]] + monomer_bonds
                dihedral = [ terminal_atom-1, terminal_atom, len(atoms), len(atoms)+1 ]
                atoms += monomer
                atoms.rotate_dihedral(dihedral, 2*numpy.pi*numpy.random.rand())
                new_terminal_atoms.append(len(atoms)-1)

                opt(atoms, bonds)

        terminal_atoms = new_terminal_atoms

    for terminal_atom in terminal_atoms:
        for branch in xrange(2):
            atoms += Atoms('H')
            bonds += [[terminal_atom, len(atoms)-1]]

    opt(atoms, bonds)
    atoms.center(10)
    return atoms

def main():
    from time import time
    #t0 = time()
    #write('POSCAR', make_pamam(4))
    #t1 = time()
    #print t1-t0
    from ase.io import read
    print 'reading structure'
    atoms = read('PAMAM-G4.xyz')
    print 'building bonds'
    bonds = connect_bonds(atoms)
    calc = VSEPR(atoms, bonds)
    atoms.set_calculator(calc)
    t0 = time()
    opt = FIRE(atoms, trajectory='opt.traj')
    opt.run(fmax=1e-3, steps=1000)
    t1 = time()
    print 'time:',t1-t0

    #atoms = Atoms('HOH')
    #bonds = [[0,1],[1,2]]
    #atoms.rattle(1.0)
    #calc = VSEPR(atoms, bonds)
    #atoms.set_calculator(calc)
    #opt = FIRE(atoms, trajectory='opt.traj')
    #opt.run(fmax=1e-3, steps=1000)

    #from ase.constraints import FixAtoms
    #mask = [ atom.symbol == 'Au' for atom in atoms ]
    #atoms.set_constraint(FixAtoms(mask=mask))

    #atoms.set_calculator(VSEPR(atoms, nonbonded=True))
    #from ase.optimize import LBFGS, FIRE; LBFGS; FIRE
    #opt = FIRE(atoms, trajectory='opt.traj')
    #opt.run(fmax=1e-3, steps=1000)


if __name__ == '__main__':
    main()
