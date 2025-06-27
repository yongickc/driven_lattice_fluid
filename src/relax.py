import numpy as np
from numba import njit
import sys
import os
from numba.typed import List
from search import condensed_search, dilute_search
from set_rate import set_rate_mat, update_mat_rate_rxn, update_mat_rate_displace

# Run simulation without collecting fluxes
@njit
def relax(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta,rng):
    mat_nB, mat_nBI, mat_rate = set_rate_mat(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta)

    for count in range(100*(lattice != 0).sum()):
        cumul = mat_rate.sum()
        rnd = rng.random()*cumul
        v = np.searchsorted(mat_rate.cumsum(),rnd,side="right")
        y = v // Lx
        x = v % Lx

        update_rxn = 0
        update_x = 0
        update_y = 0
        molecule_type = lattice[y,x] # 1 = B; 0 = S; -1 = I
        nB = mat_nB[y,x]
        nBI = mat_nBI[y,x]

        rnd = rng.random() * mat_rate[y,x]
        # Diffusion
        if (molecule_type == 1 and rnd < D*np.exp(e*nB)*(4-nBI)/4) or\
                (molecule_type == -1 and rnd < D*(4-nBI)/4):
            while True:
                rnd1 = rng.random()
                if rnd1 < 0.25:
                    if lattice[(y+1)%Ly,x] == 0:
                        update_y = 1
                        break
                elif rnd1 < 0.5:
                    if lattice[y,(x+1)%Lx] == 0:
                        update_x = 1
                        break
                elif rnd1 < 0.75:
                    if lattice[(y-1)%Ly,x] == 0:
                        update_y = -1
                        break
                else:
                    if lattice[y,(x-1)%Lx] == 0:
                        update_x = -1
                        break
            lattice[(y+update_y)%Ly,(x+update_x)%Lx] = molecule_type
            lattice[y,x] = 0
            mat_rate[y,x] = 0
        # B->I
        elif molecule_type == 1:
            lattice[y,x] = -1
            update_rxn = -1
            mat_rate[y,x] = kIB[nB] + D*(4-nBI)/4

        # I->B
        elif molecule_type == -1:
            lattice[y,x] = 1
            update_rxn = 1
            mat_rate[y,x] = kBI[nB] + D*np.exp(e*nB)*(4-nBI)/4

        # Update rate matrices after reaction
        if update_rxn:
            update_mat_rate_rxn(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,update_rxn,Ly,Lx,y,x)

        # Update rate matrices after molecule displacement
        elif update_x != 0 or update_y != 0:
            update_mat_rate_displace(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,molecule_type,update_y,update_x,Ly,Lx,y,x)

    return lattice

