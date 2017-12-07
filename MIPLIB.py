import numpy as np
import scipy as sp
import scipy.sparse					# To handle sparse data
import scipy.sparse.linalg 			# To perform linear algebra operation on sparse data
import copy


from Cuts import *
from ReprVars import *
from CPLEXInterface import *

################################
# PRIMARY INPUTS - CHECK THESE #
################################

filename = 'enlight13'
# Put the .mps file in a folder named MIPLIB or renamed the below appropriately
prefix = './MIPLIB/'
postfix = '.mps'

verbose = True # Verbose solving of cplex models. Keeping this true can help in  debugging.

################################

C_org = cplex.Cplex()
C_org.read(prefix+filename+postfix)

# Convert the problem to standard form - function defined in CPLEXInterface.py
C = Cplex2StdCplex(prefix+filename+postfix, MIP  = True, verbose = verbose)
# Store the integrality constraints before removing them to solve it as an LP relaxation
cont_var = np.array([1 if i=='C' else 0 for i in C.variables.get_types()])
int_var = 1-cont_var
# Remove the integrality constraints
C.set_problem_type(C.problem_type.LP)
# Store a copy of the file in standard form
C.write(prefix+filename+'_std'+postfix)

# Solve the problem in CPLEX with all presolving turned off - function defined in CPLEXInterface.py
LPSolution = getfromCPLEX(C, verbose=verbose, ForceSolve=True)
print(LPSolution["Objective"])
Tableaux = LPSolution["Tableaux"]


# intRows is a function defined in Cuts.py to identify which integrality constraints are NOT met
bad_variable = np.where(intRows(
	LPSolution["Solution"][LPSolution["Basic"]], 
	int_var[LPSolution["Basic"]]
	)
)
# bad_variable should be = 165. 
# The problem in hand enlight13 is such that the integrality constraint of only one variable is violated
# in the LP relaxation.
# The 165th variable has an integrality constraint that is violated
# in the LP relaxation


# Now, this means 165th row (actually 166th - we count from 0) of the tableaux should have 
# non-integer values. (If all elements in the 165th row are integers, 
# then that is a proof that the problem is infeasible! But we know enlight13 is 
# IP feasible and bounded.)
Tabl165 = Tableaux[165, :]
# In fact it is a vector of integers!
print (np.linalg.norm(np.remainder(Tabl165,1))) # The norm is 0.

# Interestingly, it is the 168th row that has the required properties
Tabl168 = Tableaux[168, :]
nonzero = np.where(Tabl168) # This is the location of non-zero elements of Tabl168

print(nonzero, Tabl168[nonzero]) 
print('In fact if I compute the non-basics with my personal simplex implementation, ')
print('this 168th row matches exactly with the expected 165 th row.')
print('Making me believe CPLEX is doing some presolving etc.')

####### Taking an alternate path to compute simplex tableaux. (Note now we are importing scipy.sparse classes)

# Get the set of basic variables from CPLEX
Basics = np.array(C.solution.basis.get_basis()[0])

# Total number of variables in the problem
nVar = np.size(Basics)

# Indices corresponding to basic and non-basic variables
B_in = np.where(Basics)[0]
N_in = np.array(list(set(range(nVar))-set(B_in)))


# The set of constraints in the problem
rows = C.linear_constraints.get_rows()

# The RHS of the Ax = b constraints
b = C.linear_constraints.get_rhs()

# initializing row_index/colum_index/data to initialize scipy.sparse matrix
# Remember CPLEX returns the rows in a sparse form - (cplex.sparse form but not the scipy.sparse form )
# We stick to scipy.sparse as that has better library functions, documentation etc.
row_ind = np.array([]).reshape(0,)
col_ind = np.array([]).reshape(0,)
data = np.array([]).reshape(0,)
for i,row in zip(range(len(rows)), rows):
    t1 = np.array(row.ind)
    row_ind = np.concatenate((row_ind, np.zeros(t1.shape) + i))
    col_ind = np.concatenate((col_ind, t1))
    t2 = np.array(row.val)
    data = np.concatenate((data, t2))
# Creating the sparse matrix
Aeq = sp.sparse.csc_matrix((data, (row_ind, col_ind)))
# Basic matrix
B = Aeq[:, B_in]
# Non basic matrix
N = Aeq[:, N_in]

# Tableau of non-basic variables is just B^{-1}N. So we are solving the linear system B*tTabl = N
tTabl = sp.sparse.linalg.spsolve(B, N)
# The basic solution is B^{-1}b. So we are solving the linear system B*x_B = b
x_B = sp.sparse.linalg.spsolve(B, b)

print(np.where(x_B ==0.5)[0]) # This should be 165... 165 the basic element is non-integer

print(tTabl[165,:]) # 165th row is not an integer row now!

