import cplex
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

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

    
def Cplex2StdCplex(filename, MIP = False, verbose = 0, MIPobject = False):
    """
    C = Cplex2StdCplex(filename, MIP = False, verbose = 0, MIPobject = False)
    Reads in a filename containing details for mixed integer problem not necessarily in a standard form.
    Returns a CPLEX object with the problem in a standard form min cTTx s.t. Ax = b; x >= 0 and integer constraints.
    If MIP = False, then an LP relaxed object is returned otherwise an MIP object is returned.
    If MIPobject = False, it is assumed the input is a file name. If True, then the 
    input is considered to be a Cplex object. Note that the changes caused here could alter the input Cplex object
    as cplex object is passed as reference in python
    """
    if MIPobject:
        M = filename
    else:
        M = cplex.Cplex()
        if  verbose <= 0:
             M.set_log_stream(None)                                          # Don't print log on screen
             M.set_results_stream(None)                                      # Don't print progress on screen    
             M.set_warning_stream(None)
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
        # Sign for inequality constraints. If it is an equality, then this does not matter
        sign = 1 if senses[i]=='L' else -1
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
    if verbose > 0:
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
    if verbose > 0:
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
            if -lb[i] < cplex.infinity:
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
    if verbose > 0:
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
    if verbose > 0:
        print("Slacks added. Eq: ", eq, " Ineq: ", ineq-eq, ' Nvar: ', len(f))
    if M.objective.sense[M.objective.get_sense()] == 'maximize':
            f = [-fi for fi in f]
    M_std = cplex.Cplex()
    M_std.variables.add(obj = f)
    M_std.objective.set_sense(M_std.objective.sense.minimize)
    M_std.linear_constraints.add(rhs = beq+b, senses = ['E']*(ineq))
    M_std.linear_constraints.set_coefficients(zip(Aeqrowind+Arowind, Aeqcolind+Acolind, Aeqval+Aval))
    name = str.replace(str.replace(M.get_problem_name(), '/','_'),'.','_') # Get the name replacing any . and / signs to _
    M_std.set_problem_name(name)
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
    MIP = Cplex2Py(M)

    If Sparse is set as true, returns sparse Aeq and rest as full objects. MIP object is not returned, 
    f, Sparse_Aeq, beq, cont are returned. Note that this works only if the CPLEX object
    is a problem in standard form. 
    """
    if sparse: # Note that CPLEX object should be in standard form 
        # Objective
        f = np.array(M.objective.get_linear())
        # Integer or continuous?
        cont = [1 if i=='C' else 0 for i in M.variables.get_types()]
        # Get constraint rows as Cpelx SparseInd
        rows = M.linear_constraints.get_rows()
        row_ind = np.zeros([]).reshape(0,)
        col_ind = np.zeros([]).reshape(0,)
        data = np.zeros([]).reshape(0,)
        for i,row in zip(range(len(rows)), rows):
            t1 = np.array(row.ind)
            row_ind = np.zeros(t1.shape) + i
            col_ind = np.concatenate((col_ind, t1))
            t2 = np.array(row.val)
            data = np.concatenate((data, t2))

    else:
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


def getfromCPLEX(M, 
    solution = True,    
    objective = True,   
    tableaux = True, 
    basic = True, 
    TablNB = True,
    precission = 13, verbose = 0, ForceSolve = False):
    """
    Given a CPLEX model M
        Sol = getfromCPLEX(M,  solution = True, objective = True,   tableaux = True, basic = True, TablNB = True, precission = 13)
    returns a dictionary Sol.   
    solution    when true returns the value of the solution decision variables in the dictionary Sol
    objective   when true returns the value of the objective in the dictionary Sol
    tableaux    when true returns the final simplex tableaux with BOTH basic and non-basics
    TablNB      when true returns the non-basic tableaux calculated as B^{-1}N using scipy.sparse matrix methods. 
                This is valid only if the cplex problem is in standard form. Otherwise returned objects are meaningless.
    basic       when true returns the indices of basic variables in the optimal basis
    precission  number of decimal points to use in the simplex Tableaux
    """
    if verbose <= 0 :
        M.set_log_stream(None)                                          # Don't print log on screen
        M.set_results_stream(None)                                      # Don't print progress on screen    
        M.set_warning_stream(None)
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
    if M.solution.get_status()==0 or ForceSolve: # If the model is not already solved or if ForceSolve is set to be true
        M.solve()                  # then solve it now
    Sol = dict()    
    # Objective
    if objective: # adding the objective
        Sol["Objective"] = M.solution.get_objective_value()
    if tableaux or basic or solution or TablNB:
        # Solution
        tsol = np.array(M.solution.get_values()) # Getting the value from CPLEX and converting to np.array
        tsol = np.around(tsol, precission).reshape((np.size(tsol),1)) # rounding it off to appropriate precission and 
        if solution: # adding the solution      
            Sol["Solution"] = tsol
        if tableaux or basic or TablNB:
            # Finding the set of basic variables from the simplex tableaux
            nVar = M.variables.get_num() # Number of variables
            nCon = M.linear_constraints.get_num() # Number of constraints
            # Get the set of basic variables from CPLEX            
            h1,h2 = M.solution.basis.get_header()
            h1 = np.array(h1)
            h2 = np.array(h2)
            b1 = np.where(h1>=0)[0]
            b2 = np.where(h1<0)[0]
            B_in = np.sort(h1[b1]).copy()
            N_in = np.sort(np.array(list(set(range(nVar))-set(B_in)))) # Non basic is 1:nVar \setminus tBasic
            redundantRow = -h1[b2].copy()-1
            usefulRows = np.sort(np.array(list(set(range(nCon))-set(redundantRow)))) # All rows \setminus redundant rows
            if verbose > 0:
                print('Redundant rows: ',redundantRow)
        # Basic variables
        if basic:
            Sol["Basic"] = np.sort(B_in)
            Sol["NonBasic"] = np.sort(N_in)
        # Tableaux
        if tableaux:
            tTabl = np.array(M.solution.advanced.binvarow())
            tTabl = np.around(tTabl, precission)        
            Sol["Tableaux"] = tTabl
        if TablNB: # Valid only if the problem is in standard form. Otherwise returns garbage.
            # The set of constraints in the problem
            rows = M.linear_constraints.get_rows()
            # The RHS of the Ax = b constraints
            b = np.array(M.linear_constraints.get_rhs())
            # initializing row_index/colum_index/data to initialize scipy.sparse matrix
            # Remember CPLEX returns the rows in a sparse form - (cplex.sparse form but not the scipy.sparse form )
            # We stick to scipy.sparse as that has better library functions, documentation etc.
            row_ind = np.array([]).reshape(0,)
            col_ind = np.array([]).reshape(0,)
            data = np.array([]).reshape(0,)
            rowcount = 0
            for i,row in zip(range(len(rows)), rows):
                if i in usefulRows:
                    t1 = np.array(row.ind)
                    row_ind = np.concatenate((row_ind, np.zeros(t1.shape) + rowcount))
                    rowcount = rowcount + 1
                    col_ind = np.concatenate((col_ind, t1))
                    t2 = np.array(row.val)
                    data = np.concatenate((data, t2))
            # Creating the sparse matrix
            Aeq = sp.sparse.csc_matrix((data, (row_ind, col_ind)), shape = (nCon - redundantRow.size, nVar))
            # Basic matrix
            B = Aeq[:, B_in]
            # Non basic matrix
            N = Aeq[:, N_in]
            # Tableau of non-basic variables is just B^{-1}N. So we are solving the linear system B*tTabl = N
            tTablNB = sp.sparse.linalg.spsolve(B, N)
            # The basic solution is B^{-1}b. So we are solving the linear system B*x_B = b
            x_B = sp.sparse.linalg.spsolve(B, b[usefulRows])
            Sol["Tableaux_NB"] = np.around(tTablNB, precission)
            Sol["Sol_Basic"] = np.around(x_B, precission)
            Sol["B"] = B
            Sol["N"] = N
            Sol["b"] = b[usefulRows]
    return Sol
