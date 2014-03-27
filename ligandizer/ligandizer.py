#!/usr/bin/env python
import numpy
numpy.seterr(all='raise')
from sys import stdout
from ase import Atom
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

from vsepr import VSEPR, connect_bonds, connect_angles, connect_nonbonded

def ligand_size(ligand):
    size = 0.0
    for i in xrange(len(ligand)):
        for j in xrange(i+1, len(ligand)):
            d = ligand.get_distance(i,j)
            d = max(d, size)
    return size

def add_ligand(atoms, ligand, site, position=None):
    atoms.set_tags(0)
    ligand = ligand.copy()
    # Extract nanoparticle.
    np_idx = [ i for i in xrange(len(atoms)) if atoms[i].number > 20 ]
    nanoparticle = atoms[np_idx]

    # Pick a binding site and shift ligand above it.
    com = nanoparticle.get_center_of_mass()
    v = atoms[site].position - com
    v /= numpy.linalg.norm(v)
    ligand.center()

    displacement_length = 1.1 * ligand_size(ligand)
    if position is None:
        ligand.positions += atoms.positions[site] + displacement_length*v
    else:
        ligand.positions += position + displacement_length*v
    ligand.set_tags(1)

    # Create combined system.
    atoms += ligand

    # Construct VSEPR calculator with N-Au interaction.
    ligand_mask = [tag == 1 for tag in atoms.get_tags()]
    ligand_bonds = connect_bonds(atoms, mask=ligand_mask)
    calculator = VSEPR(atoms,
                       ligand_bonds + [[site,len(atoms)-len(ligand)]],
                       nonbonded=connect_nonbonded(atoms,
                                                   connect_angles(ligand_bonds),
                                                   r_cut=displacement_length*2))
    atoms.set_calculator(calculator)

    atoms.set_constraint(FixAtoms(indices=range(len(atoms)-len(ligand))))

    # Run optimization.
    opt = LBFGS(atoms, memory=10, maxstep=0.2)
    opt.run(fmax=1e-2)

def add_ligands(atoms,
                ligand,
                corner_sites=True,
                edge_sites=[],
                facet_100_sites=[],
                facet_111_sites=[],
                output_stream=stdout):

    ligand_counter = 1

    log = lambda s: output_stream.write(s+'\n')

    # Determine coordination numbers.
    cns = numpy.zeros(len(atoms), int)
    for i in xrange(len(atoms)):
        for j in xrange(i+1, len(atoms)):
            d = atoms.get_distance(i,j)
            if d < 3.3:
                cns[i] += 1
                cns[j] += 1

    # Identify the vertices (corners) of the polyhedron.
    vertices = []
    for i in xrange(len(atoms)):
        if cns[i] > 6:
            continue
        vertices.append(i)
        if corner_sites:
            log('ligand %3i: adding to corner %i' % (ligand_counter, i))
            add_ligand(atoms, ligand, i)
            ligand_counter += 1

    # How close are the vertices to one another? Needed to find edges.
    min_vertex_distance = None
    for i in xrange(len(vertices)):
        for j in xrange(i+1, len(vertices)):
            ii = vertices[i]
            jj = vertices[j]
            d = atoms.get_distance(ii, jj, mic=True)
            if min_vertex_distance is None:
                min_vertex_distance = d
            elif d < min_vertex_distance:
                min_vertex_distance = d

    # Edges are nearest vertex pairs.
    edges = []
    for i in xrange(len(vertices)):
        for j in xrange(i+1, len(vertices)):
            ii = vertices[i]
            jj = vertices[j]
            d = atoms.get_distance(ii, jj)
            if d < 1.2*min_vertex_distance:
                edge = [ii,jj]
                edges.append(edge)

                for dx in edge_sites:
                    v1 = atoms.positions[edge[1]] - atoms.positions[edge[0]]
                    position = atoms.positions[edge[0]]+dx*v1
                    log('ligand %3i: %.3f fractional position along edge (%3i,%3i)' % (ligand_counter, dx, edge[0], edge[1]))
                    atom = Atom('Au', position=position)
                    atoms += atom
                    idx = len(atoms)-1
                    add_ligand(atoms, ligand, idx)
                    ligand_counter += 1
                    atoms.pop(idx)

    face_111_history = []
    face_100_history = []
    for i in xrange(len(edges)):
        for j in xrange(len(edges)):
            if i==j:continue
            edge1 = set(edges[i])
            edge2 = set(edges[j])

            common_vertex = edge1 & edge2
            if len(common_vertex) == 0:
                continue
            angle = list(edge1-common_vertex) + list(common_vertex) + list(edge2-common_vertex)
            theta = atoms.get_angle(angle)*180/numpy.pi

            # 100 square face
            if abs(theta - 90.0) < 1.0:
                v1 = atoms.positions[angle[1]] - atoms.positions[angle[0]]
                v2 = atoms.positions[angle[2]] - atoms.positions[angle[1]]

                for dx,dy in facet_100_sites:
                    skip = False
                    n1 = numpy.cross(v1,v2)
                    n1 /= numpy.linalg.norm(n1)
                    for n2,angle2 in face_100_history:
                        if 1.0-numpy.abs(numpy.dot(n1,n2)) > 1e-3:
                            continue

                        p1 = atoms[angle[0]].position
                        p2 = atoms[angle2[0]].position
                        v3 = p2-p1
                        if numpy.abs(numpy.dot(v3, n1)) < 1e-3:
                            skip=True
                            break
                    if skip:
                        continue
                    face_100_history.append([n1,angle])

                    position=atoms.positions[angle[0]]+dx*v1+dy*v2

                    best = 0
                    best_distance = 100.0
                    for idx, atom in enumerate(atoms):
                        if atom.symbol != 'Au': continue
                        if cns[idx] == 12: continue
                        d = numpy.linalg.norm(atoms.positions[idx] - position)
                        if d < best_distance:
                            best_distance = d
                            best = idx
                    msg = 'ligand %3i: at (%6.3f,%6.3f) to 100 face (%3i,%3i,%3i)'
                    log(msg % (ligand_counter, dx,dy,angle[0],angle[1],angle[2]))
                    add_ligand(atoms, ligand, best, position)
                    ligand_counter += 1

            if abs(theta - 60.0) < 1.0:
                v1 = atoms.positions[angle[1]] - atoms.positions[angle[0]]
                v2 = atoms.positions[angle[2]] - atoms.positions[angle[1]]

                skip = False
                for angle2 in face_111_history:
                    if len(set(angle) & set(angle2)) == 3:
                        skip = True
                        break
                if skip:
                    continue
                face_111_history.append(angle)

                for dx,dy in facet_111_sites:

                    position=atoms.positions[angle[0]]+dx*v1+dy*v2

                    best = 0
                    best_distance = 100.0
                    for idx, atom in enumerate(atoms):
                        if atom.symbol != 'Au': continue
                        d = numpy.linalg.norm(atoms.positions[idx] - position)
                        if d < best_distance:
                            best_distance = d
                            best = idx
                    msg = 'ligand %3i: at (%6.3f,%6.3f) position to 111 face (%3i,%3i,%3i)'
                    log(msg % (ligand_counter,dx,dy,angle[0],angle[1],angle[2]))
                    add_ligand(atoms, ligand, best, position)
                    ligand_counter += 1

    atoms.center(5.0)
    atoms.set_constraint()
    return atoms
