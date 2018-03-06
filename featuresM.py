import numpy as np
import scipy as sp
import scipy.sparse					# To handle sparse data
import scipy.sparse.linalg 			# To perform linear algebra operation on sparse data
import copy
import io as  cStringIO
import re


from Cuts import *
from ReprVars import *
from CPLEXInterface import *


# Input: a CPLEX object c, and time limit (in seconds) TL for the probing
# Out: a dictionary with {feature name : feature val}
def getProbingFeatures(c, TL):
   #dictionary to be returned
   D = {}

   #sets time limit
   c.parameters.timelimit.set(TL)

   #sets up display info
   out = cStringIO.StringIO()
   c.set_results_stream(out)
   c.parameters.mip.display.set(3)

   #Solves only rootnode (INCLUDING LP!), without any cuts (but primal heuristics, etc. are all on, maybe affects the results)
   c.solve()

   #######################
   #Reads off following features: CPU times for presolving and relaxation
   #, # of constraints, variables, nonzero entries in the constraint matrix,
   #and clique table inequalities after presolving 
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
     
   D['numRowsPresolved'] = int(numRowsPresolved)
   D['numColsPresolved'] = int(numColsPresolved)
   D['numNonzerosPresolved'] = int(numNonzerosPresolved)
   D['totalPresolveTime'] = float(totalPresolveTime)
   D['totalProbingTime'] = float(totalProbingTime)
   D['cliqueTable'] = int(cliqueTable)

   #######################
   # Computes number of each of 7 different cut types, and total cuts applied
   cutNames = ["cover", "GUB_cover", "flow_cover", "clique", "fractional", "MIR", "flow_path", "disjunctive", "implied_bound", "zero_half", "multi_commodity_flow", "lift_and_project"]

   #param value in class c.solution.MIP.cut_type, indexed relative to the cutNames
   cutParamVal = [0,1,2,3,4,5,6,7,8,9,10,14]

   numCuts = [0] * len(cutNames)

   for i in range(len(cutNames)):
      numCuts[i] = c.solution.MIP.get_num_cuts(cutParamVal[i])
      D['numCuts' + cutNames[i]] = int(numCuts[i])

   numCutsTotal = sum(numCuts)
   D['numCutsTotal'] = int(numCutsTotal)

   #######################
   # Computes number of iterations and number of nodes
   numIter = c.solution.progress.get_num_iterations()
   numNodesProc = c.solution.progress.get_num_nodes_processed()
   D['numIter'] = int(numIter)
   D['numNodesProc'] = int(numNodesProc)

   # Computes relative gap (WHAT TO DO IF NO FOLUTION FOUND? Gets exception (try on enlight13))
   #relGap = c.solution.MIP.get_mip_relative_gap() 

   return D

################################
# PRIMARY INPUTS - CHECK THESE #
################################

#filename = 'example'
filename = 'enlight13'
#filename = 'gmut-75-50'
#prefix = './mps/'
prefix = './MIPLIB/'
postfix = '.mps'

#time limit used for the whole probing 
timelim = 5

c = cplex.Cplex()
c.read(prefix+filename+postfix)

D = getProbingFeatures(c, timelim)

print(D)
