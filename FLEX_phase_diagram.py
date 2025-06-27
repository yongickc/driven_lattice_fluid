import numpy as np
import sys
from scipy.optimize import root
from scipy.special import k0,k1

e = float(sys.argv[1])
rhov = float(sys.argv[2])
eta = 1.0

def set_rate(e,df,µdr,eta):
    kIB = np.full(5,eta).astype(np.float64)
    kBI = np.full(5,eta).astype(np.float64)

    for b in range(5):
        t = np.exp(e*b+df+µdr)
        if t < 1:
            kBI[b] *= t
        else:
            kIB[b] /= t
        kBI[b] += np.exp(e*b+df)
        kIB[b] += 1

    return kIB, kBI

# NOTE: Predicting the phase diagram for the flat interface case in the strongly phase-separated limit
#       in which flux and density gradient within the condensed phase is ignorable.
def solve_flat(x):
    df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3 = x
    kIB,kBI = set_rate(e,df,µdr,eta)

    Kv = kIB[0]/kBI[0]

    rhoBv = (rhoBp+rhoIp) * Kv/(1+Kv)
    rhoIv = (rhoBp+rhoIp) * 1/(1+Kv)

    Dv = Lambda/4
    xiv = np.sqrt(Dv/(kBI[0]+kIB[0]))

    jBp = Dv/xiv*(rhoBp-rhoBv)
    jIp = Dv/xiv*(rhoIp-rhoIv)

    # NOTE: Estimation of p(nB) using solid-on-solid approximation (see Saito, p.37)
    #       This was ultimately not used for Fig. 4 in the paper but the end result hardly depends on p(nB).
    #J1 = np.log(pB1/(1-pB1))
    #q = np.exp(J1/2) # Consider noneq. fluctuation?
    #q = np.exp(J/2)  # Fix. to eq.
    #p_up = q/(1+q)
    #p_nB = np.array([(1-2*p_up+2*p_up**2)/2,2*p_up*(1-p_up),(1-2*p_up+2*p_up**2)/2]) # nB = 1,2,3

    pB = np.array([pB1,pB2,pB3])
    pI = np.array([pI1,pI2,pI3])
    b = np.array([1,2,3])

    rxn_BI_nB = kBI[1:4]*pB - kIB[1:4]*pI #sBI - sIB

    jBp_FLEX_nB = Dv*(np.exp(e*b)*pB*(1-rhoBp-rhoIp) - rhoBp*(1-pB-pI))
    jIp_FLEX_nB = Dv*(pI*(1-rhoBp-rhoIp) - rhoIp*(1-pB-pI))

    pB_balance = -rxn_BI_nB - jBp_FLEX_nB
    pI_balance =  rxn_BI_nB - jIp_FLEX_nB

    jBp_FLEX = (jBp_FLEX_nB * p_nB).sum()
    jIp_FLEX = (jIp_FLEX_nB * p_nB).sum()

    return np.concatenate([pB_balance, pI_balance, [(p_nB*pB).sum()-0.5, rhoBp+rhoIp-rhov, jBp_FLEX-jBp]])

# NOTE: Use p(nB) obtained from the equilibrium instantaneous interface
p_nB = np.loadtxt("p_nB_eq.txt")[:,1]

# Initial guesses
Lambda = 10**2.0
µdr = 0.0
df = 2.8362
rhoBp, rhoIp = 0.003, 0.047
pB1, pB2, pB3 = 0.0497, 0.50, 0.99
pI1, pI2, pI3 = 0.05, 0.02, 0.001
sol = root(solve_flat,x0=(df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3),tol=10**(-18))
df_eq,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3 = sol.x

# NOTE: Figure 3 (upper)
for Lambda in [10**1.0,10**2.0,10**3.0]:
    list_µdr = np.arange(-2.25,2.26,0.01)
    list_df = []
    list_xi = []
    list_JB_xi = []
    list_dh2_ratio = []
    for µdr in list_µdr:
        sol = root(solve_flat,x0=(df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3),tol=10**(-18))
        df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3 = sol.x

        kIB,kBI = set_rate(e,df,µdr,eta)
        Dv = Lambda/4
        xi = np.sqrt(Dv/(kBI[0]+kIB[0]))

        rhoBv = rhov * kIB[0]/(kIB[0]+kBI[0])
        JB = Dv*(rhoBp-rhoBv)

        # Based on solid-on-solid approximation
        e_eff = np.log(pB1/(1-pB1))
        dh2_ratio = (np.sinh(e/4)/np.sinh(e_eff/4))**2

        list_df.append(df)
        list_xi.append(xi)
        list_JB_xi.append(JB/xi)
        list_dh2_ratio.append(dh2_ratio)

    list_df = np.array(list_df)
    list_xi = np.array(list_xi)
    list_JB_xi = np.array(list_JB_xi)
    list_dh2_ratio = np.array(list_dh2_ratio)
    
    txt = np.array([list_µdr,list_df-df_eq,list_xi,list_JB_xi,list_dh2_ratio]).T
    np.savetxt("Lambda_%.1f_flat.txt"%(np.log10(Lambda)),txt,fmt="%.2f %.15f %.15f %.15f %.15f")

# NOTE: Figure 3 (lower)
df = 2.8362
rhoBp, rhoIp = 0.003, 0.047
pB1, pB2, pB3 = 0.0497, 0.50, 0.99
pI1, pI2, pI3 = 0.05, 0.02, 0.001
for µdr in [-2.0,2.0]:
    list_log_Lambda = np.arange(0.75,4.26,0.01)
    list_df = []
    list_xi = []
    list_JB_xi = []
    list_dh2_ratio = []
    for Lambda in np.power(10,list_log_Lambda):
        sol = root(solve_flat,x0=(df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3),tol=10**(-18))
        df,rhoBp,rhoIp,pB1,pB2,pB3,pI1,pI2,pI3 = sol.x

        Dv = Lambda/4
        kIB,kBI = set_rate(e,df,µdr,eta)
        xi = np.sqrt(Dv/(kBI[0]+kIB[0]))

        rhoBv = rhov * kIB[0]/(kIB[0]+kBI[0])
        JB = Dv*(rhoBp-rhoBv)

        e_eff = np.log(pB1/(1-pB1))
        dh2_ratio = (np.sinh(e/4)/np.sinh(e_eff/4))**2

        list_df.append(df)
        list_xi.append(xi)
        list_JB_xi.append(JB/xi)
        list_dh2_ratio.append(dh2_ratio)

    list_df = np.array(list_df)
    list_xi = np.array(list_xi)
    list_JB_xi = np.array(list_JB_xi)
    list_dh2_ratio = np.array(list_dh2_ratio)

    txt = np.array([list_log_Lambda,list_df-df_eq,list_xi,list_JB_xi,list_dh2_ratio]).T
    np.savetxt("µdr_%.1f_flat.txt"%(µdr),txt,fmt="%.2f %.15f %.15f %.15f %.15f")


