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
# python runInst A0 25 10 100 15
args = sys.argv
Batch = args[1]
Nvar = int(args[2])
Ncons = int(args[3])
nInst = int(args[4])
nCuts = int(args([5]))

print("Running batch: " + Batch + " with " + str(Nvar) + " variables and " +str(Ncons) + " constraints.")
values = run_compare_root(nRows = [2,5], Num_IP=nInst, 
                    NumRounds = 5, verbose = 2, 
                    path = "/home-4/ssankar5@jhu.edu/scratch/201802/", 
                    scratch = "/home-4/ssankar5@jhu.edu/scratch/201802/scratch/",
                    Batch = Batch, Nvar = Nvar, Ncons = Ncons)

[print(i) for i in values]
