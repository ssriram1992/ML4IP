import cplex
import numpy as np
import scipy as sp

def Py2Cplex(Prob):
    """
    Given a problem (MIP object)
        min c^T x subject to
        Ax = b
        x >= 0
        x_i \in \Z if i \in intcon
    returns a CPLEX model object 
    """
    M = cplex.Cplex()
    M.objective.set_sense(M.objective.sense.minimize)
    # Note integer/continuos detail is not added as CPLEX will solve this as
    # LP and not MIP.
    # Note, CPLEX has default lb = 0, and ub = +Inf. So we are not explicitly
    # setting the bounds.
    M.variables.add(obj=Prob.f.squeeze().tolist()) #Cplex requires list input
    # Adding constraints
    nCons = np.size(Prob.beq)   
    nVar = np.size(Prob.f)
    # Writing the LHS list
    LHS =  [[range(nVar), Prob.Aeq[i,:].squeeze().tolist()] for i in range(nCons)]
    # Adding the constraints
    M.linear_constraints.add(lin_expr = LHS, senses='E'*nCons, rhs=Prob.beq.squeeze().tolist())
    return M

def File2Py(filename):
    M = cplex.Cplex()
    M.read(filename)
    return Cplex2Py(M)


def createNewMip(filename, random = True, f = None, A = None, b = None, Aeq = None, beq = None, lb = None, ub = None, cont = None):
    """
    Save a generic MIP in a file
    Not a required file. But might use to create examples, to debug and test CPLEX interface.
    """
    if random:
        Nvar = 7 # Number of variables
        Nineq = 1 # Number of inequality constraints
        Neq = 2 # Number of equality constraints
        f = np.random.randint(0,10,(Nvar,1)).astype(float)
        A = np.random.randint(-10,10,(Nineq,Nvar)).astype(float)
        b = np.random.randint(10,25,(Nineq,1)).astype(float)
        Aeq = np.random.randint(-10,10,(Neq,Nvar)).astype(float)
        beq = np.random.randint(-5,5,(Neq,1)).astype(float)
        lb = np.random.randint(-3,3,(Nvar,1)).astype(float)    
        ub = np.random.randint(10,150,(Nvar,1)).astype(float)
        cont = np.random.randint(0,2,(Nvar,))
    # Contonuous or integer?
    types =  ['C' if i else 'I' for i in cont]
    # less than equal to contraints and equality contraints
    senses = "L"*len(b)+"E"*len(beq)
    # LHS for the constraints
    rows = [
        [
            [
                j for j in range(len(f))
            ], 
            [
                A[i,j] for j in range(len(f))
            ]
        ] for i in range(len(A)) 
    ] + [
        [
            [
                j for j in range(len(f))
            ], 
            [
                Aeq[i,j] for j in range(len(f))
            ]
        ] for i in range(len(Aeq))
    ]
    # RHS for the constraints
    rhs = [i for i in b.squeeze()]+[i for i in beq.squeeze()]
    M = cplex.Cplex()
    # Maximization objective. (Note this is set opposite to the convention we use - minimization)
    # Just to ensure that the code we try to debug takes this into account
    M.objective.set_sense(M.objective.sense.maximize)
    # Adding variables, lower bound, upperbound, objective vector, integrality constraint details
    M.variables.add(obj = f.squeeze().tolist(), lb = lb.squeeze().tolist(), ub = ub.squeeze().tolist(), types = types)
    # Adding constraints
    M.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs)
    # Writing it to file
    M.write(filename)
    # Returning a tuple containing all the variables
    return (f, A, b, Aeq, beq, lb, ub, cont)

    
def Cplex2StdCplex(filename, MIP = False, verbose = False):
    M = cplex.Cplex()
    # Load file
    M.read(filename)
    # Generates the sparse, row-column-value representation of constraint matrices
    # Get the detailed form of variables
    lb = M.variables.get_lower_bounds()
    ub = M.variables.get_upper_bounds()
    f = M.objective.get_linear()
    rows = M.linear_constraints.get_rows()
    senses = M.linear_constraints.get_senses()
    rhs = M.linear_constraints.get_rhs()
    if MIP:
        integrality = M.variables.get_types()
    Nvar = len(f)
    # Initializing sparse matrix row-col-val for constraints
    # Inequality constraints - Number of equality constraints remain fixed. So ineq follows eq
    ineq = sum([(sense=='E')*1 for sense in senses])
    Arowind = []
    Acolind = []
    Aval = []
    b = []
    # Equality constraints
    Aeqrowind = []
    Aeqcolind = []
    Aeqval = []
    beq = []
    eq = 0
    # Extracting from "rows"
    # In each constraint row
    for i in range(len(rows)):
        # For each entry in the SparseInd
        for j in range(len(rows[i].ind)):
            # If it is an equality constraint
            if senses[i] == 'E':
                # Add it to Aeqrowind/colind/val
                Aeqrowind.append(eq)
                Aeqcolind.append(rows[i].ind[j])
                Aeqval.append(rows[i].val[j])
            else:
                # Else add it to Arowind/colind/val depending on <= or >= constraint
                sign = 1 if senses[i]=='L' else -1
                Arowind.append(ineq)
                Acolind.append(rows[i].ind[j])
                Aval.append(sign*rows[i].val[j])            
        # Managing RHS and count of number of equality/inequality constraints
        if senses[i] == 'E':
            eq = eq+1
            beq.append(rhs[i])        
        else:
            ineq = ineq+1
            b.append(sign*rhs[i])
    # Verbose printing
    if verbose:
        print("Constraint matrices extracted. Eq: ", eq, " Ineq: ", ineq-eq, ' Nvar: ', len(f))
    # Adding the upper bound as regular constraint
    for i in range(Nvar):
        temp = ub[i]
        if temp < cplex.infinity:
            Arowind.append(ineq)
            Acolind.append(i)
            Aval.append(1.0)
            b.append(temp)  
            ineq = ineq+1
    if verbose:
        print("Upper bound constraints added. Eq: ", eq, " Ineq: ", ineq-eq, ' Nvar: ', len(f))
    # Adding the lower bound as regular constraint
    badlb = 0
    for i in range(Nvar):
        # if lb is 0, then nothing to add
        if lb[i] == 0:
            continue
        # if lb is positive, then just add a constraint. x>=0 (the default constraint) doesn't matter
        if lb[i] > 0:
            Arowind.append(ineq)
            Acolind.append(i)
            Aval.append(-1.0)
            b.append(-lb[i])
            ineq = ineq+1
        # If lb is negative, then variable has to be written as difference between two non-negative variable
        if lb[i] < 0:
            Arowind.append(ineq)
            Acolind.append(i)
            Aval.append(-1.0)
            ineq = ineq+1
            b.append(-lb[i])
            # repeating the previous occurences in ineq matrix
            t1 = np.array(Arowind)
            t2 = np.array(Acolind)
            t3 = np.array(Aval)
            s1 = (np.where(t2 == i)[0])
            Arowind = Arowind + t1[s1].tolist()
            Acolind = Acolind + [Nvar + badlb]*s1.size
            Aval = Aval + [-val for val in t3[s1]]
            # repeating the previous occurences in eq matrix
            t1 = np.array(Aeqrowind)
            t2 = np.array(Aeqcolind)
            t3 = np.array(Aeqval)
            s1 = (np.where(t2 == i)[0])
            Aeqrowind = Aeqrowind + t1[s1].tolist()
            Aeqcolind = Aeqcolind + [Nvar + badlb]*s1.size
            Aeqval = Aeqval + [-val for val in t3[s1]]
            # Extending f vector
            f.append(-f[i])
            if MIP:
                integrality.append(integrality[i])
            badlb = badlb+1
    if verbose:
        print("Lower bound constraints added. Eq: ", eq, " Ineq: ", ineq-eq, ' Nvar: ', len(f))
    Nvar = len(f)
    slack = 0
    for i in range(eq,ineq):
        Arowind.append(i)
        Acolind.append(Nvar+slack)
        Aval.append(1.0)
        slack = slack+1
        f.append(0)
        if MIP:
            integrality.append('C')
    if verbose:
        print("Slacks added. Eq: ", eq, " Ineq: ", ineq-eq, ' Nvar: ', len(f))
    if M.objective.sense[M.objective.get_sense()] == 'maximize':
            f = [-fi for fi in f]
    M_std = cplex.Cplex()
    M_std.variables.add(obj = f)
    M_std.objective.set_sense(M_std.objective.sense.minimize)
    M_std.linear_constraints.add(rhs = beq+b, senses = ['E']*(ineq))
    M_std.linear_constraints.set_coefficients(zip(Arowind+Aeqrowind, Acolind+Aeqcolind, Aval+Aeqval))
    if MIP:
        M_std.variables.set_types(zip(range(len(f)), integrality))
    return M_std


def Cplex2Py(M, sparse=False):
    """
    Given a model M in cplex converts into numpy 
    understandable arrays and a model of the form
        min c^T x subject to 
        A x <= b
        Aeq x = beq
        lb <= x <= ub
        x_i \in \Z if i \in intcon
    [c, A, b, Aeq, beq, lb, ub, intcon] = Cplex2Py(M)

    If the flag MIPobj = True is given then, it returns MIP object 
    """
    # Bound constraints
    lb = np.array(M.variables.get_lower_bounds())
    ub = np.array(M.variables.get_upper_bounds())
    # Objective
    f = np.array(M.objective.get_linear())
    # if objective is maximization, then reverse the sign of the objective
    if M.objective.sense[M.objective.get_sense()] == 'maximize':
        f = -f
    # Get constraint rows as Cpelx SparseInd
    rows = M.linear_constraints.get_rows()
    # Details whether it is = or <= or >=
    senses = M.linear_constraints.get_senses()
    # Integer or continuous?
    cont = [1 if i=='C' else 0 for i in M.variables.get_types()]
    # Full matrix to store constraints
    AllCons = np.zeros((0,len(f)))
    # Converting cplex SparseInd to numpy.array()
    for row in rows:
        t1 = np.array(row.ind)
        t2 = np.array(row.val)
        t3 = np.zeros((1,len(f)))
        t3[0,t1] =t2
        AllCons = np.concatenate((AllCons, t3))
    eq = [sense=='E' for sense in senses]
    less = [sense=='L' for sense in senses]
    great = [sense=='G' for sense in senses]
    # Dividing them based on constraint sense (=/<=/>=)
    Aeq = AllCons[eq,:]
    A = np.concatenate((AllCons[less,:], - AllCons[great,:]))
    # RHS of constraints
    rhs = np.array(M.linear_constraints.get_rhs())
    beq = np.array(rhs[eq])
    b = np.array(np.concatenate((rhs[less],-rhs[great])))
    # Returning MIP object
    return MIP(
        form = 2,
        data = {
            "f":f,
            "Aeq":Aeq,
            "beq":beq,
            "cont":cont,
            "A":A,
            "b":b,
            "lb":lb,
            "ub":ub
        }
    )


def getfromCPLEX(M, solution = True,    objective = True,   tableaux = True, basic = True, precission = 13, verbose = False, ForceSolve = False):
    """
    Given a CPLEX model M
        Sol = getfromCPLEX(M,  solution = True, objective = True,   tableaux = True, basic = True, precission = 13)
    returns a dictionary Sol.   
    solution    when true returns the value of the solution decision variables in the dictionary Sol
    objective   when true returns the value of the objective in the dictionary Sol
    tableaux    when true returns the final simplex tableaux with BOTH basic and non-basics
    basic       when true returns the indices of basic variables in the optimal basis
    precission  number of decimal points to use in the simplex Tableaux
    """
    if not verbose:
        M.set_log_stream(None)                                          # Don't print log on screen
        M.set_results_stream(None)                                      # Don't print progress on screen    
        M.parameters.simplex.display.set(2)
    # First let us turn on all presolve procedures and run 
    # Straightforward Simplex on the primal problem 
    # For an exact description of each of these, check
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.studio.help/pdf/paramcplex.pdf
    M.parameters.lpmethod.set(M.parameters.lpmethod.values.primal)  # Solve the primal problem
    M.parameters.preprocessing.dual.set(-1)                     # Don't use dual
    M.parameters.preprocessing.presolve.set(0)
    M.parameters.preprocessing.numpass.set(0)                 # Don't pass it to "other" presolves
    M.parameters.preprocessing.fill.set(0)                    # Don't aggregate variables
    M.parameters.preprocessing.linear.set(0)                  # Turn off linear reductions
    M.parameters.preprocessing.aggregator.set(0)              # Don't use aggregator
    M.parameters.preprocessing.reduce.set(0)                  # Don't use primal/dual reduction
    M.presolve.presolve(M.presolve.method.none)                     # Turn off any presolving
    if M.solution.get_status()==0 or ForceSolve: # If the model is not already solved
        M.solve()                  # then solve it now
    Sol = dict()    
    # Objective
    if objective: # adding the objective
        Sol["Objective"] = M.solution.get_objective_value()
    if tableaux or basic or solution:
        # Solution
        tsol = np.array(M.solution.get_values()) # Getting the value from CPLEX and converting to np.array
        tsol = np.around(tsol, precission).reshape((np.size(tsol),1)) # rounding it off to appropriate precission and 
        if solution: # adding the solution      
            Sol["Solution"] = tsol
        if tableaux or basic:
            # Tableaux
            tTabl = np.array(M.solution.advanced.binvarow())
            tTabl = np.around(tTabl, precission)        
            if tableaux:
                Sol["Tableaux"] = tTabl
            # Basic variables
            if basic:
                # Finding the set of basic variables from the simplex tableaux
                nVar = M.variables.get_num() # Number of variables
                nCon = M.linear_constraints.get_num() # Number of constraints
                SureBasics = np.where(tsol!=0)[0] # Non-zero variables are definitely basics
                if np.size(SureBasics) == nCon: # Number of non-zeros equal the number of constraints. So no degeneracy
                    tBasic = SureBasics.copy()
                else: # There is degeneracy in optimal basis
                    newBasicCount = nCon - np.size(SureBasics) # Number of basic variables that take the value 0
                    ProbableBasics = np.where(tsol==0)[0] # Indices of variables that have value 0
                    reducedBasic = tTabl[:, SureBasics] # The Basic matrix obtained from current choice of basic variables. At the end, this should be an identity matrix
                    # Note reducedBasic is a tall matrix. We have to add columns to make it square
                    rowsObtained = np.sum(reducedBasic, axis=1) # Have obtained "1" along these rows already in the identity matrix
                    for i in ProbableBasics: 
                        # In each of the variable with zero, check if the column in tTabl is 0 everywhere, except for a 1 in a place where
                        # rowsObtained is 0
                        temp = tTabl[:,i] # i-th column in the tableaux
                        if ( np.size(np.where(temp==0)[0]) == nCon -1 and # column has zeros everywhere except a single coordinate
                             np.sum(temp) == 1 and # The non-zero row in temp has the value 1
                             np.sum(np.any((temp, rowsObtained),axis=0)) == np.sum(rowsObtained) + 1 # The "1" in temp is in a location different from the rowsObtained
                            ): # then this column can be added to the Basic matrix
                            SureBasics = np.concatenate((SureBasics, [i]), axis=0) # Add the current variable index to sure basics                          
                            reducedBasic = tTabl[:, SureBasics] # Update reducedBasic
                            rowsObtained = np.sum(reducedBasic, axis=1)  # Update rowsObtained
                        if np.size(SureBasics) == nCon: # Number of non-zeros equal the number of constraints. So degeneracy resolved
                            break # Just so there is no necessity to check the condition for all the remaining columns
                    tBasic = np.sort(SureBasics.copy())
                Sol["Basic"] = tBasic
                tNonBasic = np.array(list(set(np.arange(nVar)) - set(tBasic))) # Non basic is 1:nVar \setminus tBasic
                Sol["NonBasic"] = tNonBasic
    return Sol