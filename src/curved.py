import numpy as np
from numba import njit
import sys
import os
from multiprocessing import Process
from numba.typed import List
from relax import relax
from search import condensed_search, dilute_search
from set_rate import set_rate, set_rate_mat, update_mat_rate_rxn, update_mat_rate_displace

# TODO: Make lattice into a class someday; numba support is experimental at the moment.

@njit
def run(lattice,Ly,Lx,e,df,µdr,D,eta,thr,rng):
    # NOTE: Set Ly = Lx = L for single droplet simulations; extension should be straightforward
    #L = min(Ly,Lx)

    kIB, kBI = set_rate(e,df,µdr,eta)
    mat_nB, mat_nBI, mat_rate = set_rate_mat(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta)

    mask = np.full((Ly,Lx),False) # Mark far-field lattice sites

    T = 0.0
    tot_rhoB = np.zeros(L,dtype=np.float64)
    tot_rhoI = np.zeros(L,dtype=np.float64)

    jB = np.zeros((3,L),dtype=np.float64)
    jI = np.zeros((3,L),dtype=np.float64)

    NBI = np.zeros((3,L),dtype=np.float64)
    NIB = np.zeros((3,L),dtype=np.float64)
    
    entropy = np.zeros((3,Ly,Lx),dtype=np.float64)

    # Far-field density 
    tot_N_far_molecule = 0.0
    N_far_molecule = 0.0
    for y in range(L):
        for x in range(L):
            if (y-Ly//2)**2 + (x-Lx//2)**2 >= thr**2:
                mask[y,x] = True
                if lattice[y,x] != 0:
                    N_far_molecule += 1

    _,droplet = condensed_search(lattice,Ly,Lx)
    dilute = dilute_search(droplet,Ly,Lx)

    # NOTE: Realign the droplet every 100 sweeps of kMC events
    for count in range(100*(lattice != 0).sum()):
        cumul = mat_rate.sum()
        rn = rng.random()*cumul
        v = np.searchsorted(mat_rate.cumsum(),rn,side="right")
        y = v // Lx
        x = v % Lx

        dt = -np.log(1.0-rng.random())/cumul
        T += dt
        tot_N_far_molecule += dt * N_far_molecule

        # NOTE: density profiles are measured along the axes passing through the CoM of the droplet
        tot_rhoB += dt * (lattice[L//2] == 1)
        tot_rhoB += dt * (lattice[:,L//2] == 1)
        tot_rhoI += dt * (lattice[L//2] == -1)
        tot_rhoI += dt * (lattice[:,L//2] == -1)

        update_rxn = 0
        update_x = 0
        update_y = 0
        molecule_type = lattice[y,x] # 1 = B; 0 = S; -1 = I
        nB = mat_nB[y,x]
        nBI = mat_nBI[y,x]

        near_droplet = droplet[y,x]
        near_dilute = dilute[y,x]
        for j,i in ((-1,0),(1,0),(0,-1),(0,1)):
            near_droplet |= droplet[(y+j)%Ly,(x+i)%Lx]
            near_dilute |= dilute[(y+j)%Ly,(x+i)%Lx]
        if (not near_droplet) and near_dilute:
            in_where = 2 # inside the dilute phase
        elif near_droplet and near_dilute:
            in_where = 1 # on the interface
        else:
            in_where = 0 # inside the droplet - includes non-B molecules within

        rn = rng.random() * mat_rate[y,x]
        # Diffusion
        if (molecule_type == 1 and rn < D*np.exp(e*nB)*(4-nBI)/4) or\
                (molecule_type == -1 and rn < D*(4-nBI)/4):
            while True:
                rn = rng.random()
                if rn < 0.25:
                    if lattice[(y+1)%Ly,x] == 0:
                        update_y = 1
                        break
                elif rn < 0.5:
                    if lattice[y,(x+1)%Lx] == 0:
                        update_x = 1
                        break
                elif rn < 0.75:
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

            if mask[y,x] and (not mask[(y+update_y)%Ly,(x+update_x)%Lx]):
                N_far_molecule -= 1
            elif (not mask[y,x]) and mask[(y+update_y)%Ly,(x+update_x)%Lx]:
                N_far_molecule += 1

            for j,i in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                near_droplet |= droplet[(y+update_y+j)%Ly,(x+update_x+i)%Lx]
                near_dilute |= dilute[(y+update_y+j)%Ly,(x+update_x+i)%Lx]
            if (not near_droplet) and near_dilute:
                in_where = 2 # inside the dilute phase
            elif near_droplet and near_dilute:
                in_where = 1 # on the interface
            else:
                in_where = 0 # inside the droplet - includes non-B molecules within

            # NOTE: Only compute fluxes along the axes passing through the CoM of the droplet
            #       in accommodation of the underlying 4-fold symmetry
            if (x == Lx//2 and update_y != 0) or (y == Ly//2 and update_x != 0):

                # NOTE: jB[y] is the +y-direction flux between y and y+1
                if x == Lx//2:
                    direction = int((update_y-1)/2)
                    if molecule_type == 1:
                        jB[in_where,(y+direction)%L] += update_y
                    else:
                        jI[in_where,(y+direction)%L] += update_y
                else:
                    direction = int((update_x-1)/2)
                    if molecule_type == 1:
                        jB[in_where,(x+direction)%L] += update_x
                    else:
                        jI[in_where,(x+direction)%L] += update_x

            if molecule_type == 1:
                # -1: Because the B-molecule now has evacuated the original lattice site 
                nB_1 = mat_nB[(y+update_y)%Ly,(x+update_x)%Lx] - 1
                S = e*(nB-nB_1)/2 # log-ratio of the forward and backward displacement rates

                # Entropy production is equally attributed to the lattice sites involved.
                entropy[in_where,y,x] += S
                entropy[in_where,(y+update_y)%L,(x+update_x)%L] += S
        
        else:
            S = 0.0
            rn = rng.random()
            # B -> I
            if molecule_type == 1:
                lattice[y,x] = -1
                update_rxn = -1
                mat_rate[y,x] = kIB[nB] + D*(4-nBI)/4
                if x == Lx//2:
                    NBI[in_where,y] += 1
                elif y == Ly//2:
                    NBI[in_where,x] += 1

                if rn < np.exp(e*nB+df) / kBI[nB]:
                    S = e*nB+df
                else:
                    S = e*nB+df+µdr

            # I -> B
            else:
                lattice[y,x] = 1
                update_rxn = 1
                mat_rate[y,x] = kBI[nB] + D*np.exp(e*nB)*(4-nBI)/4

                if x == Lx//2:
                    NIB[in_where,y] += 1
                elif y == Ly//2:
                    NIB[in_where,x] += 1

                if rn < 1.0 / kIB[nB]:
                    S = -(e*nB+df)
                else:
                    S = -(e*nB+df+µdr)

            entropy[in_where,y,x] += S

        if near_droplet:
            _,droplet = condensed_search(lattice,Ly,Lx)
            dilute = dilute_search(droplet,Ly,Lx)

        # Update for reaction
        if update_rxn:
            update_mat_rate_rxn(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,update_rxn,Ly,Lx,y,x)

        # Update for diffusion
        elif update_x != 0 or update_y != 0:
            update_mat_rate_displace(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,molecule_type,update_y,update_x,Ly,Lx,y,x)

    return tot_N_far_molecule, tot_rhoB, tot_rhoI, jB, jI, NBI, NIB, entropy, T, lattice

def kmc(L,e,df,µdr,D,eta,index,tot_particle,thr,R):
    home = os.getcwd() # TODO: Change to your own choice of directory
    rng = np.random.default_rng()

    # NOTE: Set Ly = Lx = L for single droplet simulations; extension should be straightforward
    Ly = L
    Lx = L
    # Initial configuration
    lattice = np.full((Ly,Lx),0).astype(np.int8)
    for y in range(Ly):
        for x in range(Lx):
            if (y-Ly//2)**2 + (x-Lx//2)**2 <= R**2:
                lattice[y,x] = 1

    kIB,kBI = set_rate(e,df,µdr,eta)
    n_dil_molecule = 0
    while n_dil_molecule < init_n_dil_molecule:
        y = rng.integers(Ly)
        x = rng.integers(Lx)
        if lattice[y,x] == 0:
            rn = rng.random()
            b = 0
            for i,j in ((-1,0),(1,0),(0,1),(0,-1)):
                if lattice[(y+j)%Ly,(x+i)%Lx] == 1:
                    b += 1
            if rn < kBI[b] / (kBI[b] + kIB[b]):
                lattice[y,x] = -1
            else:
                lattice[y,x] = 1
            n_dil_molecule += 1
    
    # Relax to a steady state
    list_sz = []
    for i in range(5000):
        lattice = relax(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta,rng)
        sz,droplet = condensed_search(lattice,Ly,Lx)
        mx = np.where(droplet == True)[1].mean()
        my = np.where(droplet == True)[0].mean()
        lattice = np.roll(lattice,round(Ly//2-my),axis=0)
        lattice = np.roll(lattice,round(Lx//2-mx),axis=1)
        list_sz.append(sz)
        np.save("%s/lattice_%i.npy"%(home,index),lattice)
        np.save("%s/list_sz_%i.npy"%(home,index),list_sz)
        print("#relax %i %i %i"%(index,i,sz),flush=True)

    lattice = np.load("%s/lattice_%i.npy"%(home,index))
    list_sz = np.load("%s/list_sz_%i.npy"%(home,index)).tolist()

    mask = 0
    for y in range(Ly):
        for x in range(Lx):
            if (y-Ly//2)**2 + (x-Lx//2)**2 >= thr**2:
                mask += 1

    totT = 0.0
    tot_N_far_molecule = 0.0
    tot_rhoB = np.zeros(L,dtype=np.float64)
    tot_rhoI = np.zeros(L,dtype=np.float64)

    tot_jB = np.zeros((3,L),dtype=np.float64)
    tot_jI = np.zeros((3,L),dtype=np.float64)

    tot_NBI = np.zeros((3,L),dtype=np.float64)
    tot_NIB = np.zeros((3,L),dtype=np.float64)

    tot_entropy = np.zeros((3,Ly,Lx),dtype=np.float64)

    # NOTE: Realign the CoM of the droplet to the center of the system every 100 kMC sweeps of events
    for i in range(0,10000):
        N_far_molecule, rhoB, rhoI, jB, jI, NBI, NIB, entropy, T, lattice  = run(lattice,Ly,Lx,e,df,µdr,D,eta,thr,rng)

        totT += T

        tot_N_far_molecule += N_far_molecule
        far_field_density = tot_N_far_molecule/totT/mask

        tot_rhoB += rhoB
        tot_rhoI += rhoI
        tot_jB += jB
        tot_jI += jI
        tot_NBI += NBI
        tot_NIB += NIB
        tot_entropy += entropy

        sz,droplet = condensed_search(lattice,Ly,Lx)
        mx = np.where(droplet == True)[1].mean()
        my = np.where(droplet == True)[0].mean()
        lattice = np.roll(lattice,round(Ly//2-my),axis=0)
        lattice = np.roll(lattice,round(Lx//2-mx),axis=1)
        list_sz.append(sz)

        np.save("%s/lattice_%i.npy"%(home,index),lattice)
        np.save("%s/list_sz_%i.npy"%(home,index),list_sz)

        # NOTE: Divide by 2 for x and y-axes
        np.save("%s/rhoB_%i.npy"%(home,index),tot_rhoB/totT/2)
        np.save("%s/rhoI_%i.npy"%(home,index),tot_rhoI/totT/2)

        np.save("%s/jB_%i.npy"%(home,index),tot_jB/totT/2)
        np.save("%s/jI_%i.npy"%(home,index),tot_jI/totT/2)

        np.save("%s/NBI_%i.npy"%(home,index),tot_NBI/totT/2)
        np.save("%s/NIB_%i.npy"%(home,index),tot_NIB/totT/2)

        np.save("%s/entropy_%i.npy"%(home,index),tot_entropy/totT)

        print(index,i,sz,totT,far_field_density,flush=True)

if __name__ == "__main__":
    L = int(sys.argv[1])
    e = float(sys.argv[2])
    df = float(sys.argv[3])
    µdr = float(sys.argv[4])
    D = float(sys.argv[5]) # NOTE: D here is \Lambda in the paper
    eta = float(sys.argv[6])
    init_n_dil_molecule = int(sys.argv[7]) # Control the dilute phase density
    thr = int(sys.argv[8]) # Far-field distance threshold
    start = int(sys.argv[9])
    nodes = int(sys.argv[10])
    R = float(sys.argv[11]) # Initial droplet radius
    
    jobs = [Process(target=kmc, args=(L,e,df,µdr,D,eta,start+i,init_n_dil_molecule,thr,R)) for i in range(nodes)]
    for job in jobs: job.start()
    for job in jobs: job.join()

