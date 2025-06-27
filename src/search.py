import numpy as np
from numba import njit
import sys
import os
from numba.typed import List

# BFS search of the condensed droplet defined as 
# the largest B-molecule cluster connected via nearest-neighboring interaction
@njit
def condensed_search(lattice,Ly,Lx):
    queue = List()
    max_sz = 30 # Smallest size assumed for efficient search
    mark = (lattice == 1)
    cluster = np.full((Ly,Lx),False)
    # Assume CoM doesn't diffuse fast so that at least one of 5x5 lattice site
    # at the center of the system belongs to the condensed phase
    for y in range(Ly//2-2,Ly//2+2+1):
        for x in range(Lx//2-2,Lx//2+2+1):
            if mark[y,x]:
                sz = 1
                cluster1 = np.full((Ly,Lx),False)
                cluster1[y,x] = True
                mark[y,x] = False
                queue.append((y,x))
                while queue:
                    y1,x1 = queue.pop()
                    for j,i in ((-1,0),(1,0),(0,1),(0,-1)):
                        y2,x2 = (y1+j)%Ly,(x1+i)%Lx
                        if mark[y2,x2]:
                            sz += 1
                            cluster1[y2,x2] = True
                            mark[y2,x2] = False
                            queue.append((y2,x2))
                if sz > max_sz:
                    cluster = np.copy(cluster1)
                    max_sz = sz

    return max_sz, cluster

# BFS search of the dilute phase, defined as the complement of the condensed phase
@njit
def dilute_search(condensed,Ly,Lx):
    queue = List()
    max_sz = 0
    mark = np.logical_not(condensed)
    cluster = np.full((Ly,Lx),False)
    for y in range(Ly):
        for x in range(Lx):
            if mark[y,x]:
                sz = 1
                cluster1 = np.full((Ly,Lx),False)
                cluster1[y,x] = True
                queue.append((y,x))
                while queue:
                    y1,x1 = queue.pop()
                    for j,i in ((-1,0),(1,0),(0,1),(0,-1)):
                        y2,x2 = (y1+j)%Ly,(x1+i)%Lx
                        if mark[y2,x2]:
                            sz += 1
                            cluster1[y2,x2] = True
                            mark[y2,x2] = False
                            queue.append((y2,x2))
                if sz > max_sz:
                    cluster = np.copy(cluster1)
                    max_sz = sz

    return cluster

