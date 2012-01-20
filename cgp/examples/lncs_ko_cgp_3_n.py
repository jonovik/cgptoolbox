#!/usr/bin/env python
# Time limit HH:mm:ss
#PBS -lwalltime=2:00:00
##PBS -lwalltime=00:05:00
# Join standard error into output
#PBS -j oe
# Mail on abort, begin, end
#PBS -m abe
#PBS -lpmem=2000MB
"""
cGP computations on SERCA knockout LNCS model. Submit with: python -m lncs_ko_cgp_3_n
"""

__version__ = "$Revision: 2278 $"
id = "$Id: lncs_cgp_3_n.py 2278 2010-09-08 16:31:35Z jonvi $"

from ap_cvode import Ff
from lncs_cgp import *
from utils.arrayjob import *  # chdir to PBS_O_WORKDIR, where job was submitted

# OPTIONS
set_NID(1440) # ID range will be 0, 1, ..., get_NID()-1

genenames=("stim_period Ko Nao Cao P_CaL g_Na vmup_init V_max_NCX g_Kr "
        "g_K1 g_Kto_f g_ClCa Km_Cl".split())[:10]

scenario = "ko" # Alternatives: ff ko bl6

li = Ff(maxsteps=500000, chunksize=20000, reltol=1e-10)
li.pr[:] = li.scenarios[scenario]["p"]
li.y[:] = li.scenarios[scenario]["y"].item()

cgp = Lncs_cgp(Genotype(msd_first=False, 
    n=np.rec.fromrecords([[3] * len(genenames)], names=genenames)), 
    max_nap=3000, li=li)

if __name__ == "__main__":
    arun(cgp.setup, cgp.work, savetohdf, makeplots, loglevel="DEBUG")
