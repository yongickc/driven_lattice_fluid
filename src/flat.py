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
def run(lattice,Ly,Lx,e,df,µdr,D,eta,rng):
    kIB, kBI = set_rate(e,df,µdr,eta)
    mat_nB, mat_nBI, mat_rate = set_rate_mat(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta)

    T = 0.0
    
    # Concentration profiles
    rhoB = np.zeros(Lx,dtype=np.float64)
    rhoI = np.zeros(Lx,dtype=np.float64)
    distB = (lattice == 1).sum(axis=0).astype(np.float64)
    distI = (lattice == -1).sum(axis=0).astype(np.float64)

    # Diffusive flux densities
    jB = np.zeros((3,Lx),dtype=np.float64)
    jI = np.zeros((3,Lx),dtype=np.float64)

    # Reactive flux densities
    NBI = np.zeros((3,Lx),dtype=np.float64)
    NIB = np.zeros((3,Lx),dtype=np.float64)

    # Entropy production rate density
    entropy = np.zeros((3,Ly,Lx),dtype=np.float64)

    _,condensed = condensed_search(lattice,Ly,Lx)
    dilute = dilute_search(condensed,Ly,Lx)

    # NOTE: Realign the condensed phase every 100 sweeps of kMC events
    for count in range(100*(lattice != 0).sum()):
        cumul = mat_rate.sum()
        rn = rng.random()*cumul
        v = np.searchsorted(mat_rate.cumsum(),rn,side="right")
        y = v // Lx
        x = v % Lx

        dt = -np.log(1.0-rng.random())/cumul
        rhoB += dt * distB
        rhoI += dt * distI
        T += dt
    
        update_rxn = 0
        update_x = 0
        update_y = 0
        molecule_type = lattice[y,x] #1 = B; 0 = S; -1 = I
        nB = mat_nB[y,x]
        nBI = mat_nBI[y,x]

        near_condensed = condensed[y,x]
        near_dilute = dilute[y,x]
        for j,i in ((-1,0),(1,0),(0,-1),(0,1)):
            near_condensed |= condensed[(y+j)%Ly,(x+i)%Lx]
            near_dilute |= dilute[(y+j)%Ly,(x+i)%Lx]

        if (not near_condensed) and near_dilute:
            in_where = 2 # inside the dilute phase
        elif near_condensed and near_dilute:
            in_where = 1 # on the interface
        else:
            in_where = 0 # inside the condensed phase - includes non B-molecules within

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
            
            # NOTE: Flux is considered interfacial when either of the two lattice sites is an interfacial site.
            for j,i in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                near_condensed |= condensed[(y+update_y+j)%Ly,(x+update_x+i)%Lx]
                near_dilute |= dilute[(y+update_y+j)%Ly,(x+update_x+i)%Lx]
            if (not near_condensed) and near_dilute:
                in_where = 2 # inside the dilute phase
            elif near_condensed and near_dilute:
                in_where = 1 # on the interface
            else:
                in_where = 0 # inside the condensed phase - includes non-B molecules within

            lattice[(y+update_y)%Ly,(x+update_x)%Lx] = molecule_type
            lattice[y,x] = 0
            mat_rate[y,x] = 0

            # Update concentration Profile
            if molecule_type == 1:
                distB[x] -= 1
                distB[(x+update_x)%Lx] += 1
            else:
                distI[x] -= 1
                distI[(x+update_x)%Lx] += 1

            # Update diffusive flux density
            # NOTE: jB[x] is the B-molecule diffusive flux between x and x+1 in +x-direction
            if update_x != 0:
                direction = int((update_x-1)/2)
                if molecule_type == 1:
                    jB[in_where,(x+direction)%Lx] += update_x
                else:
                    jI[in_where,(x+direction)%Lx] += update_x

            # Entropy production rate density
            # Only count B-molecule since the displacement rate of I-molecules is always constant
            if molecule_type == 1:
                # -1: Because B-molecule has moved from the original site
                nB_1 = mat_nB[(y+update_y)%Ly,(x+update_x)%Lx] -1 
                S = (nB-nB_1)*e/2 # log-ratio of forward and backward displacement rates
                
                # Entropy production is equally attributed to the lattice sites involved.
                entropy[in_where,y,x] += S
                entropy[in_where,(y+update_y)%Ly,(x+update_x)%Lx] += S

        # Reaction
        else:
            S = 0.0
            rn = rng.random()
            # B -> I
            if molecule_type == 1:
                distB[x] -= 1
                distI[x] += 1
                NBI[in_where,x] += 1
                if rn < np.exp(e*nB+df) / kBI[nB]:
                    S = e*nB+df
                else:
                    S = e*nB+df+µdr
                lattice[y,x] = -1
                update_rxn = -1
                mat_rate[y,x] = kIB[nB] + D*(4-nBI)/4
            # I -> B
            else:
                distB[x] += 1
                distI[x] -= 1
                NIB[in_where,x] += 1
                if rn < 1.0 / kIB[nB]:
                    S = -(e*nB+df)
                else:
                    S = -(e*nB+df+µdr)
                lattice[y,x] = 1
                update_rxn = 1
                mat_rate[y,x] = kBI[nB] + D*np.exp(e*nB)*(4-nBI)/4

            # Entropy production rate density
            entropy[in_where,y,x] += S

        if near_condensed:
            _,condensed = condensed_search(lattice,Ly,Lx)
            dilute = dilute_search(condensed,Ly,Lx)

        # Update for reaction
        if update_rxn:
            update_mat_rate_rxn(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,update_rxn,Ly,Lx,y,x)

        # Update for diffusion
        elif update_x != 0 or update_y != 0:
            update_mat_rate_displace(kIB,kBI,e,D,lattice,mat_nB,mat_nBI,mat_rate,molecule_type,update_y,update_x,Ly,Lx,y,x)

    return rhoB, rhoI, jB, jI, NBI, NIB, entropy, T, lattice

def kmc(Ly,Lx,e,df,µdr,D,eta,index):
    home = os.getcwd() # TODO: Change to your own choice of directory
    rng = np.random.default_rng()

    # Prepare the dilute phase with a random initial configuration
    vapor = np.full((Ly,Lx-Ly),0).astype(np.int8)
    kIB,kBI = set_rate(e,df,µdr,eta)
    n_particle = 0
    while n_particle < 0.05*Ly*(Lx-Ly):
        x = rng.integers(Lx-Ly)
        y = rng.integers(Ly)
        if vapor[y,x] == 0:
            b = 0
            for j,i in ((-1,0),(1,0),(0,-1),(0,1)):
                if vapor[(y+j)%Ly,(x+i)%(Lx-Ly)] == 1:
                    b += 1
            rn = rng.random()
            if rn < kBI[b]/(kBI[b]+kIB[b]):
                vapor[y,x] = -1
            else:
                vapor[y,x] = 1
            n_particle += 1

    for i in range(100):
        vapor = relax(kIB,kBI,vapor,Ly,Lx-Ly,e,df,µdr,D,eta,rng)

    # Place a Ly x Ly block of B-molecules at the center of the system and relax the system
    lattice = np.full((Ly,Lx),1).astype(np.int8)
    lattice[:,Ly:] = np.copy(vapor)
    lattice = np.roll(lattice,(Lx-Ly)//2,axis=1)

    for i in range(5000):
        lattice = relax(kIB,kBI,lattice,Ly,Lx,e,df,µdr,D,eta,rng)
        np.save("%s/lat_%i.npy"%(home,index),lattice)
        print("#relax %i %i"%(index,i),flush=True)

    lattice = np.load("%s/lattice_%i.npy"%(home,index))

    tot_rhoB = np.zeros(Lx,dtype=np.float64)
    tot_rhoI = np.zeros(Lx,dtype=np.float64)
    tot_jB = np.zeros((3,Lx),dtype=np.float64)
    tot_jI = np.zeros((3,Lx),dtype=np.float64)
    tot_NBI = np.zeros((3,Lx),dtype=np.float64)
    tot_NIB = np.zeros((3,Lx),dtype=np.float64)
    tot_entropy = np.zeros((3,Ly,Lx),dtype=np.float64)
    totT = 0.0

    tot_list_left = []
    tot_list_right = []
    
    # Run the simulation and realign the condensed phase every 100 sweeps of kMC events
    for i in range(10000):
        _,condensed = condensed_search(lattice,Ly,Lx)
        mean_y = np.where(condensed == True)[1].astype(np.float64).mean()
        lattice = np.roll(lattice,round(Lx//2-mean_y),axis=1)

        rhoB, rhoI, jB, jI, NBI, NIB, entropy, T, lattice = run(lattice,Ly,Lx,e,df,µdr,D,eta,rng)

        totT += T

        tot_rhoB += rhoB
        tot_rhoI += rhoI
        np.save("%s/rhoB_%i.npy"%(home,index),tot_rhoB/totT/Ly)
        np.save("%s/rhoI_%i.npy"%(home,index),tot_rhoI/totT/Ly)

        tot_jB += jB
        tot_jI += jI
        np.save("%s/jB_%i.npy"%(home,index),tot_jB/totT/Ly)
        np.save("%s/jI_%i.npy"%(home,index),tot_jI/totT/Ly)

        tot_NBI += NBI
        tot_NIB += NIB
        np.save("%s/NBI_%i.npy"%(home,index),tot_NBI/totT/Ly)
        np.save("%s/NIB_%i.npy"%(home,index),tot_NIB/totT/Ly)

        tot_entropy += entropy
        np.save("%s/entropy_%i.npy"%(home,index),tot_entropy/totT)

        np.save("%s/lattice_%i.npy"%(home,index),lattice)

        # Save Interface configuration
        list_left = []
        for y in range(Ly):
            for x in range(Lx):
                if (not condensed[y,x]) and condensed[y,(x+1)%Lx]:
                    list_left.append(x)
                    break
        tot_list_left.append(list_left)
        np.save("%s/left_%i.npy"%(home,index),tot_list_left)

        list_right = []
        for y in range(Ly):
            for x in range(Lx-1,0,-1):
                if (not condensed[y,x]) and condensed[y,(x-1)%Lx]:
                    list_right.append(x)
                    break
        tot_list_right.append(list_right)
        np.save("%s/right_%i.npy"%(home,index),tot_list_right)

        print(index,i,totT,flush=True)

if __name__ == "__main__":
    Ly = int(sys.argv[1])
    Lx = int(sys.argv[2])
    e = float(sys.argv[3])
    df = float(sys.argv[4])
    µdr = float(sys.argv[5])
    D = float(sys.argv[6]) # NOTE: D here is \Lambda in the paper
    eta = float(sys.argv[7])
    nodes = int(sys.argv[8])
    start = int(sys.argv[9])
    
    jobs = [Process(target=kmc, args=(Ly,Lx,e,df,µdr,D,eta,start+i)) for i in range(nodes)]
    for job in jobs: job.start()
    for job in jobs: job.join()

