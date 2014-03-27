#!/usr/bin/env python
from collections import namedtuple
from ase.data import covalent_radii, atomic_numbers, vdw_radii
from ase.structure import molecule; molecule
from ase.io import read; read
import numpy
numpy.seterr(all='raise')

class VSEPR:

    def __init__(self, atoms, bonds, angles=[], nonbonded=[]):
        self.bonds = bonds
        self.angles = connect_angles(bonds)

        if nonbonded == 'auto':
            self.nonbonded = connect_nonbonded(atoms, self.angles, 3.0)
        else:
            self.nonbonded = nonbonded

        self.atom_types = determine_atom_types(atoms, bonds)
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
        u, f = force_field(self.atoms,
                           self.atom_types,
                           self.bonds,
                           self.angles,
                           self.nonbonded)
        self.u = u
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
        'H' : 0.00086728,
        'C' : 0.0052037,
        'O' : 0.0086728,
        'N' : 0.0069383,
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

def force_field(atoms, atom_types, bonds, angles, nonbonded):
    total_energy = 0.0
    forces = numpy.zeros((len(atoms), 3))

    for bond in bonds:
        i,j = bond
        r = atoms.get_distance(i, j, mic=True)
        r_eq, k_r = bond_stretch_parameter(atom_types[i], atom_types[j])

        total_energy += 0.5 * k_r * (r - r_eq)**2
        force = k_r * (r - r_eq)

        diff_r = atoms.positions[i] - atoms.positions[j]
        diff_r = diff_r/numpy.linalg.norm(diff_r)
        forces[i] -= force*diff_r
        forces[j] += force*diff_r

    for angle in angles:
        i,j,k = angle

        theta_eq, k_theta = angle_bending_parameter(atom_types[i],
                                                    atom_types[j],
                                                    atom_types[k])

        diff_r1 = atoms.positions[i] - atoms.positions[j]
        r1_sq = numpy.dot(diff_r1, diff_r1)
        r1 = numpy.sqrt(r1_sq)

        diff_r2 = atoms.positions[k] - atoms.positions[j]
        r2_sq = numpy.dot(diff_r2, diff_r2)
        r2 = numpy.sqrt(r2_sq)

        c = numpy.dot(diff_r1, diff_r2)
        c /= r1*r2

        if (c > 1.0): c = 1.0
        if (c < - 1.0): c = -1.0

        s = numpy.sqrt(1.0 - c**2)
        if (s < 1e-3): s = 1e-3
        s = 1.0/s

        dtheta = numpy.arccos(c) - theta_eq
        tk = k_theta * dtheta

        total_energy += tk*dtheta

        a = -2.0 * tk * s
        a11 = a*c / r1_sq
        a12 = -a / (r1*r2)
        a22 = a*c / r2_sq

        f1 = numpy.zeros(3)
        f3 = numpy.zeros(3)

        f1[0] = a11*diff_r1[0] + a12*diff_r2[0]
        f1[1] = a11*diff_r1[1] + a12*diff_r2[1]
        f1[2] = a11*diff_r1[2] + a12*diff_r2[2]
        f3[0] = a22*diff_r2[0] + a12*diff_r1[0];
        f3[1] = a22*diff_r2[1] + a12*diff_r1[1];
        f3[2] = a22*diff_r2[2] + a12*diff_r1[2];

        forces[i] += f1
        forces[j] -= f1 + f3
        forces[k] += f3

    for i,j in nonbonded:
        r = atoms.get_distance(i, j, mic=True)

        sigma, epsilon = nonbonded_parameter(atom_types[i], atom_types[j])

        total_energy += 4*epsilon*((sigma/r)**12-(sigma/r)**6)
        force = 4*epsilon*(12*(sigma/r)**13-6*(sigma/r)**7)


        r_diff = atoms.positions[i] - atoms.positions[j]
        forces[i] += force*r_diff/r
        forces[j] -= force*r_diff/r

    return total_energy, forces

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
            #print atoms[i]
            #for bond in bonds:
            #    if i in bond: print bond
            #raise ValueError
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

def connect_nonbonded(atoms, angles, r_cut=20.0, mask=None):
    nonbonded = []
    for i in xrange(len(atoms)):
        if not mask is None and mask[i] == False: continue
        for j in xrange(i+1, len(atoms)):
            if atoms[i].number > 20 and atoms[j].number > 20:
                continue
            skip = False
            for angle in angles:
                if i in angle and j in angle:
                    skip = True
                    break
            if skip: continue
            r = atoms.get_distance(i,j,mic=True)
            if r < r_cut:
                nonbonded.append([i,j])
    return nonbonded

def connect_analysis(atoms):
    print 'Connect analysis for %s' % atoms.get_chemical_formula('reduce')

    bonds = connect_bonds(atoms)
    angles = connect_angles(bonds)
    nonbonded = connect_nonbonded(atoms, angles)
    atom_types = determine_atom_types(atoms, bonds)

    from pprint import pprint
    print 'bond pairs:', len(bonds)
    pprint(bonds)
    print
    print 'angles triples:', len(angles)
    pprint(angles)
    print
    print 'nonbonded pairs:', len(nonbonded)
    pprint(nonbonded)
    print
    print 'atom types:'
    for i,atom_type in enumerate(atom_types):
        print '%4i: Element: %2s type: %s' % (i, atom_type.symbol, atom_type.hybridization)

    print

    print 'Force Field Parameters'
    print
    print 'Bond Terms'
    already_seen = []
    for bond in bonds:
        i,j = bond
        if {atoms[i].symbol, atoms[j].symbol} in already_seen:
            continue
        else:
            already_seen.append({atoms[i].symbol, atoms[j].symbol})
        r_eq, k_bond = bond_stretch_parameter(atom_types[i], atom_types[j])
        print '%s-%s: r_eq=%.3f Ang k_bond=%.3f eV/Ang^2' % \
            (atoms[i].symbol, atoms[j].symbol, r_eq, k_bond)
    print
    print 'Angle Terms'
    already_seen = []
    for angle in angles:
        i,j,k = angle
        if {atoms[i].symbol, atoms[j].symbol, atoms[k].symbol} in already_seen:
            continue
        else:
            already_seen.append({atoms[i].symbol, atoms[j].symbol, atoms[k].symbol})
        theta_eq, k_theta= angle_bending_parameter(atom_types[i], atom_types[j], atom_types[k])
        print '%s-%s-%s: theta_eq=%.3f deg k_theta=%.3e eV/deg^2' % \
            (atoms[i].symbol, atoms[j].symbol, atoms[k].symbol,
             theta_eq*180/numpy.pi, k_theta*(numpy.pi/180)**2)

def demo_H2O():
    atoms = molecule('H2O')
    connect_analysis(atoms)

def demo_C2H4():
    atoms = molecule('C2H4')
    connect_analysis(atoms)

def demo_benzene():
    atoms = read('benzene.xyz')
    connect_analysis(atoms)

def demo_pamam():
    atoms = read('PAMAM-monomer.xyz')
    connect_analysis(atoms)

def demo_Au147_pamam():
    atoms = read('CONTCAR')
    connect_analysis(atoms)

def main():
    #demo_C2H4()
    #demo_benzene()
    #demo_pamam()
    demo_Au147_pamam()

    #from ase.constraints import FixAtoms
    #mask = [ atom.symbol == 'Au' for atom in atoms ]
    #atoms.set_constraint(FixAtoms(mask=mask))

    #atoms.set_calculator(VSEPR(atoms, nonbonded=True))
    #from ase.optimize import LBFGS, FIRE; LBFGS; FIRE
    #opt = FIRE(atoms, trajectory='opt.traj')
    #opt.run(fmax=1e-3, steps=1000)


if __name__ == '__main__':
    main()
