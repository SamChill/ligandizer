#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct atom_t {
    double x,y,z;
    double fx,fy,fz;
}atom_t;

typedef struct bond_t {
    struct atom_t *atom1, *atom2;
    double r_eq, k;
}bond_t;

typedef struct nonbond_t {
    struct atom_t *atom1, *atom2;
    double r_vdw, sigma;
}nonbond_t;

typedef struct angle_t {
    struct atom_t *atom1, *atom2, *atom3;
    double theta_eq, k;
}angle_t;

typedef enum {HARMONIC, LENNARDJONES} style_t;

double bond_terms(atom_t *atoms, bond_t *bonds, int n_bonds, style_t style)
{
    double energy = 0.0;
    for (int bond_index=0;bond_index<n_bonds;bond_index++) {
        atom_t *atom1 = bonds[bond_index].atom1;
        atom_t *atom2 = bonds[bond_index].atom2;

        double dx = atom1->x - atom2->x;
        double dy = atom1->y - atom2->y;
        double dz = atom1->z - atom2->z;

        double r = sqrt(dx*dx+dy*dy+dz*dz);
        double r_eq = bonds[bond_index].r_eq;
        double k_stretch = bonds[bond_index].k;

        double dr = (r-r_eq);
        double fmag=0;
        if (style == HARMONIC) {
            energy += 0.5 * k_stretch * dr*dr;
            fmag = k_stretch*dr;
        }else if (style == LENNARDJONES) {
            double r_ratio = r_eq/r;
            double a = r_ratio*r_ratio*r_ratio*r_ratio*r_ratio*r_ratio;
            double b = 4*k_stretch*a;
            
            energy += b*(a-1);

            fmag=-6*b/r*(2*a-1);
        }
        atom1->fx -= fmag*dx/r;
        atom1->fy -= fmag*dy/r;
        atom1->fz -= fmag*dz/r;
        atom2->fx += fmag*dx/r;
        atom2->fy += fmag*dy/r;
        atom2->fz += fmag*dz/r;
    }

    return energy;
}

double angle_terms(atom_t *atoms, angle_t *angles, int n_angles)
{
    double energy = 0.0;
    for (int angle_index=0;angle_index<n_angles;angle_index++) {
        atom_t *atom1 = angles[angle_index].atom1;
        atom_t *atom2 = angles[angle_index].atom2;
        atom_t *atom3 = angles[angle_index].atom3;

        double dx1 = atom1->x - atom2->x;
        double dy1 = atom1->y - atom2->y;
        double dz1 = atom1->z - atom2->z;

        double rsq1 = dx1*dx1+dy1*dy1+dz1*dz1;
        double r1 = sqrt(rsq1);

        double dx2 = atom3->x - atom2->x;
        double dy2 = atom3->y - atom2->y;
        double dz2 = atom3->z - atom2->z;

        double rsq2 = dx2*dx2+dy2*dy2+dz2*dz2;
        double r2 = sqrt(rsq2);

        double c = dx1*dx2 + dy1*dy2 + dz1*dz2;
        c /= r1*r2;

        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        double s = sqrt(1.0 - c*c);
        if (s < 1e-3) s = 1e-3;
        s = 1.0/s;

        double theta_eq = angles[angle_index].theta_eq;
        double dtheta = acos(c) - theta_eq;


        double k_bend = angles[angle_index].theta_eq;
        energy += 0.5*k_bend*dtheta*dtheta;

        double a = -2.0 * k_bend * dtheta * s;
        double a11 = a*c / rsq1;
        double a12 = -a / (r1*r2);
        double a22 = a*c / rsq2;


        double f1[3], f3[3];
        f1[0] = a11*dx1 + a12*dx2;
        f1[1] = a11*dy1 + a12*dy2;
        f1[2] = a11*dz1 + a12*dz2;
        f3[0] = a22*dx2 + a12*dx1;
        f3[1] = a22*dy2 + a12*dy1;
        f3[2] = a22*dz2 + a12*dz1;

        //printf("r1: %.3f r2: %.3f\n", r1, r2);
        //printf("a: %f a11: %f a12: %f a22: %f\n", a, a11, a12, a22);
        //printf("f1: %f %f %f\n", f1[0], f1[1], f1[2]);
        //printf("f2: %f %f %f\n", f3[0], f3[1], f3[2]);
        //printf("dx1: %f dx2: %f\n", dx1, dx2);
        //printf("dy1: %f dy2: %f\n", dy1, dy2);
        //printf("dz1: %f dz2: %f\n", dz1, dz2);

        atom1->fx += f1[0];
        atom1->fy += f1[1];
        atom1->fz += f1[2];
        atom2->fx -= f1[0] + f3[0];
        atom2->fy -= f1[1] + f3[1];
        atom2->fz -= f1[2] + f3[2];
        atom3->fx += f3[0];
        atom3->fy += f3[1];
        atom3->fz += f3[2];
    }

    return energy;
}

void vsepr(double *positions, double *forces, int num_atoms, double *energy,
           int *bonds_list, int num_bonds, double *k_bonds, double *r_bonds,
           int *angles_list, int num_angles, double *k_angles, double *theta_angles,
           int *nonbonded_list, int num_nonbonds, double *sigmas, double *r_vdws)
{
    atom_t *atoms    = malloc(sizeof(atom_t)*num_atoms);
    bond_t *bonds    = malloc(sizeof(bond_t)*num_bonds);
    angle_t *angles  = malloc(sizeof(angle_t)*num_angles);
    bond_t *nonbonds = malloc(sizeof(bond_t)*num_nonbonds);

    for (int i=0;i<num_atoms;i++) {
        atoms[i].x = positions[3*i+0];
        atoms[i].y = positions[3*i+1];
        atoms[i].z = positions[3*i+2];
        //printf("atom  %3i: %.3f %.3f %.3f\n", i, atoms[i].x, atoms[i].y, atoms[i].z);
        atoms[i].fx = 0.0;
        atoms[i].fy = 0.0;
        atoms[i].fz = 0.0;
    }

    for (int i=0;i<num_bonds;i++) {
        bonds[i].atom1 = &atoms[bonds_list[2*i+0]];
        bonds[i].atom2 = &atoms[bonds_list[2*i+1]];
        //printf("bond  %3i: (%i,%i) %.3f %.3f\n", i, bonds_list[2*i], bonds_list[2*i+1], r_bonds[i], k_bonds[i]);
        bonds[i].r_eq = r_bonds[i];
        bonds[i].k = k_bonds[i];
    }

    for (int i=0;i<num_angles;i++) {
        angles[i].atom1 = &atoms[angles_list[3*i+0]];
        angles[i].atom2 = &atoms[angles_list[3*i+1]];
        angles[i].atom3 = &atoms[angles_list[3*i+2]];
        //printf("angle %3i: (%i,%i,%i) %.3f %.3f\n", angles_list[3*i], angles_list[3*i+1], angles_list[3*i+2], i, theta_angles[i], k_angles[i]);
        angles[i].theta_eq = theta_angles[i];
        angles[i].k = k_angles[i];
    }

    for (int i=0;i<num_nonbonds;i++) {
        nonbonds[i].atom1 = &atoms[nonbonded_list[2*i+0]];
        nonbonds[i].atom2 = &atoms[nonbonded_list[2*i+1]];
        nonbonds[i].r_eq = r_vdws[i];
        nonbonds[i].k = sigmas[i];
    }

    double bond_energy    = bond_terms(atoms, bonds, num_bonds, HARMONIC);
    double angle_energy   = angle_terms(atoms, angles, num_angles);
    double nonbond_energy = 0.0;
    if (num_nonbonds > 0) {
        nonbond_energy = bond_terms(atoms, nonbonds, num_nonbonds, LENNARDJONES);
    }

    *energy = bond_energy + angle_energy + nonbond_energy;

    for (int i=0;i<num_atoms;i++) {
        forces[3*i+0] = atoms[i].fx;
        forces[3*i+1] = atoms[i].fy;
        forces[3*i+2] = atoms[i].fz;
    }
}
