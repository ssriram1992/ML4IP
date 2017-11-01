import cplex

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
    # Note integer/continuos detail is not ed as CPLEX will solve this as
    # LP and not MIP.
    # Note, CPLEX has default lb = 0, and ub = +Inf. So we are not explicitly
    # setting the bounds.
    M.variables.add(obj=Prob.f.squeeze().tolist()) #Cplex requires list input
    # Adding constraints
    nCons = np.size(Prob.beq)   
    nVar = np.size(Prob.f)
    # Writing the LHS list
    LHS =  [[range(nVar), Afin[i,:].squeeze().tolist()] for i in range(nCons)]
    # Adding the constraints
    M.linear_constraints.add(lin_expr = LHS, senses='E'*nCons, rhs=Prob.beq.squeeze().tolist())
    

def Cplex2Py(M, MIPobj = False):
    """
    Given a model M in cplex converts into numpy 
    understandable arrays and a model of the form
        min c^T x subject to 
        A x <= b
        Aeq x = beq
        lb <= x <= ub
        x_i \in \Z if i \in intcon
    [c, A, b, Aeq, beq, lb, ub, intcon] = Cplex2Py(M)

    If the flag MIPobj = True is given then, it return 

    """

def getfromCPLEX(M, solution = True,    objective = True,   tableaux = True, basic = True, precission = 13):
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
    M.parameters.lpmethod.set(M.parameters.lpmethod.values.primal)  # Solve the primal problem
    M.presolve.presolve(M.presolve.method.none)                     # Turn off any presolving
    if M.solution.get_status()==0: # If the model is not already solved
        M.solve()                  # then solve it now
    Sol = dict()    
    # Objective
    if objective: # adding the objective
        Sol["objective"] = M.solution.get_objective_value()
    if tableaux or basic or solution:
        # Solution
        tsol = np.array(M.solution.get_values()) # Getting the value from CPLEX and converting to np.array
        tsol = np.around(tsol, precission).reshape((np.size(tsol),1)) # rounding it off to appropriate precission and 
        if solution: # adding the solution      
            Sol["solution"] = tsol
        if tableaux or basic:
            # Tableaux
            tTabl = np.array(M.solution.advanced.binvarow())
            tTabl = np.around(tTabl, precission)        
            if tableaux:
                Sol["tableaux"] = tTabl
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