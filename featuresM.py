import numpy as np
import scipy as sp
import scipy.sparse					# To handle sparse data
import scipy.sparse.linalg 			# To perform linear algebra operation on sparse data
import copy
import cStringIO
import re


from Cuts import *
from ReprVars import *
from CPLEXInterface import *

################################
# PRIMARY INPUTS - CHECK THESE #
################################

filename = 'example'
filename = 'enlight13'
filename = 'gmut-75-50'
prefix = './mps/'
prefix = './MIPLIB/'
postfix = '.mps'

################################

#time limit in secs 
timelim = 5

c = cplex.Cplex()
c.read(prefix+filename+postfix)
c.parameters.mip.limits.nodes.set(0)

#Disables all cuts (Gomory fractional cuts are not included in the eachcutlimit)
#c.parameters.mip.limits.eachcutlimit.set(0)
#c.parameters.mip.cuts.gomory.set(-1)
c.parameters.timelimit.set(timelim)

#sets up display info
out = cStringIO.StringIO()
c.set_results_stream(out)
c.parameters.mip.display.set(3)

#Solves only rootnode (INCLUDING LP!), without any cuts (but primal heuristics, etc. are all on, maybe affects the results)
c.solve()

#######################
#######################
#From s we can read off: 48-53. Presolving features: CPU times for presolving and relaxation, # of constraints, variables, nonzero entries in the constraint matrix, and clique table inequalities after presolving 
s = out.getvalue()

totalPresolveTime = 0.0
totalProbingTime = 0.0
cliqueTable = 0
numRowsPresolved = -1.0
numColsPresolved = -1.0
numNonzerosPresolved = -1.0

lines = s.splitlines()
linesIter = iter(lines)

for line in linesIter:  
   if line.startswith("Presolve time"):
      ret = re.search("Presolve time = ([0-9\.]+)", line)
      totalPresolveTime += float(ret.group(1))
   elif line.startswith("Probing time"):
      ret = re.search("Probing time = ([0-9\.]+)", line)
      totalProbingTime += float(ret.group(1))
   elif line.startswith("Reduced MIP"):
      ret = re.search("Reduced MIP has ([0-9]+) rows, ([0-9]+) columns, and ([0-9]+)", line)
      numRowsPresolved = ret.group(1)
      numColsPresolved = ret.group(2)
      numNonzerosPresolved = ret.group(3)

      #skips next line (if needed to get it, just assign nextLine = next(....) 
      next(linesIter, None)
   elif line.startswith("Clique table"):
      ret = re.search("Clique table members: ([0-9]+)", line)
      cliqueTable = ret.group(1)
      
print("numRowsPresolved: " + str(numRowsPresolved))
print("numColsPresolved: " + str(numColsPresolved))
print("numNonzerosPresolved:" + str(numNonzerosPresolved))
print("Total presolve: " + str(totalPresolveTime))
print("Total probing: " + str(totalProbingTime))
print("Clique: " + str(cliqueTable))

#######################
#######################
#54-61. Probing cut usage features: number of each of 7 different cut types, and total cuts applied
cutNames = ["cover", "GUB_cover", "flow_cover", "clique", "fractional", "MIR", "flow_path", "disjunctive", "implied_bound", "zero_half", "multi_commodity_flow", "lift_and_project"]

#param value in class c.solution.MIP.cut_type, indexed relative to the cutNames
cutParamVal = [0,1,2,3,4,5,6,7,8,9,10,14]

numCuts = [0] * len(cutNames)

for i in xrange(len(cutNames)):
   numCuts[i] = c.solution.MIP.get_num_cuts(cutParamVal[i])
   print(cutNames[i] + ": " + str(numCuts[i]))

numTotalCuts = sum(numCuts)

######################
######################
numIter = c.solution.progress.get_num_iterations()
numNodesProc = c.solution.progress.get_num_nodes_processed()

#Throws exception if no solution has been found
#relGap = c.solution.MIP.get_mip_relative_gap()
#print(relGap)



#Other parts of the API that could be useful:
#c.presolve.presolve(c.presolve.method.none)
#c.set_log_stream("log.txt")

