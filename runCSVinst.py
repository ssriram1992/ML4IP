import cplex
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

from Cuts import *
from ReprVars import *
from CPLEXInterface import *
from runMIPLIB import *

import sys

# Usage 
args = sys.argv

Fnames = args[1]
F_naam = args[1].split("_")[0]
path = "/home-4/ssankar5@jhu.edu/scratch/201804/"+F_naam+"/"

values = run_compare_root_csv(ListFile = Fnames, NumRounds = 1, nRows = [2]*20 + [5]*20 + [10]*20, 
                solPrefix = "full_", solSuffix = "_more",
                nCuts = 10, path = path, verbose = 1, scratch = "/home-4/ssankar5@jhu.edu/scratch/201804/scratch/", doX = False, doGX = True) 



# print("Running batch: " + Batch + " with " + str(Nvar) + " variables and " +str(Ncons) + " constraints.")
# values = run_compare_root_rat(nRows = [2,5,10], Num_IP=nInst, 
                    # NumRounds = 5, verbose = 1, 
                    # path = "/home-4/ssankar5@jhu.edu/scratch/201804/", 
                    # scratch = "/home-4/ssankar5@jhu.edu/scratch/201804/scratch/",
                    # Batch = Batch+"_rat", Nvar = Nvar, Ncons = Ncons)
# 
# print("Running batch: " + Batch + " with " + str(Nvar) + " variables and " +str(Ncons) + " constraints.")
# values = run_compare_root(nRows = [2,5,10], Num_IP=nInst, 
                    # NumRounds = 5, verbose = 1, 
                    # path = "/home-4/ssankar5@jhu.edu/scratch/201804/", 
                    # scratch = "/home-4/ssankar5@jhu.edu/scratch/201804/scratch/",
                    # Batch = Batch, Nvar = Nvar, Ncons = Ncons)
# 
[print(i) for i in values]
