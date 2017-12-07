import numpy as np
import scipy as sp
import scipy.sparse

import scipy.sparse.linalg
import copy

np.set_printoptions(precision=3)

from Cuts import *
from ReprVars import *
from CPLEXInterface import *

def run_compare_root(Batch = "A", Num_IP = 10, Nvar = 25, Ncons = 10, NumRounds = 10, nRows = [2,3,5,10], nCuts = 10, verbose = False):    
    values = []
    problem = 0
    while problem < Num_IP:
        A = np.random.uniform(-5,6,size = (Ncons, Nvar)) 
        # Generate a non-negative vector for variable values
        temp = np.round(60*np.random.rand(Nvar).reshape(Nvar,1))/10 
        # Choosing b this way ensures LP feasibility
        b = A.dot(temp)
        f = np.random.randint(-5,10, size = (Nvar,1))  
        cont = np.random.randint(0,2, size = (Nvar,))
        M = MIP(form = 1, data = {
            'Aeq':A,
            'beq':b,
            'f':f,
            'cont':cont
        })
        name = Batch + "_" + str(problem+1)
        _,v = compare_root_problem(M, NumRounds, name = name, nRows = nRows, nCuts = nCuts, verbose = verbose)
        if v is None:
            if verbose or True:
                print(name+" repeating with error")
            continue
        M.write(name)
        np.savetxt(name + "_Sol.csv", np.array(v), delimiter = ',')
        values.append(v)
        print(name+" completed")
        problem = problem + 1
    np.savetxt(Batch + "_Sol.csv", np.array(values), delimiter = ',')
    return values


def compare_root_problem(M, NumRounds, nRows = [2,3,5,10], nBad = 1, nCuts = 10, verbose = False, name = "_"):
    """
    M is an "our" MIP object
    """
    # Create Cplex Object
    C = Py2Cplex(M)
    cont = M.cont
    C.set_problem_name(name)
    values = []
    names = []
    # Solve the LP relaxation
    LPS = getfromCPLEX(C, ForceSolve=True)  
    if C.solution.get_status_string() != "optimal": # If LP optimal is not being obtained
        if verbose or True:
            print (name + ' '+C.solution.get_status_string())
        return None, None  
    names.append('LP')
    values.append(LPS["Objective"])
    # Add GMI cuts for pure integer version
    C_GMI_p,_ = addUserCut(C, cont*0, LPS, verbose = verbose)
    C_GMI_p.solve()
    names.append('GMI_p')
    values.append(C_GMI_p.solution.get_objective_value())
    # Add GMI cuts for mixed integer version
    C_GMI_m,_ = addUserCut(C, cont, LPS, verbose = verbose)
    C_GMI_m.solve()
    names.append('GMI_m')
    values.append(C_GMI_m.solution.get_objective_value())
    for row_ct in nRows:
        ############################################
        ############ REGULAR X-POLYTOPE ############
        ############################################
        # Pure integer
        # Add X cuts        
        C_X_p, best_X_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_X_p_'+str(row_ct))
        values.append(C_X_p.solution.get_objective_value())
        # Add XG cuts
        C_XG_p, best_XG_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_XG_p_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_XG_p = addUserCut(C, cont*0, LPS, "X", 
                            cutDetails={'ans':best_X_p},
                            )
        C2_XG_p.solve()
        values.append(np.maximum(
                C2_XG_p.solution.get_objective_value(),
                C_XG_p.solution.get_objective_value()
            ))
        # Mixed integer
        # X cut
        C_X_m, best_X_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_X_m_'+str(row_ct))
        values.append(C_X_m.solution.get_objective_value())
        # Add XG cuts
        C_XG_m, best_XG_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_XG_m_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_XG_m = addUserCut(C, cont, LPS, "X", 
                            cutDetails={'ans':best_X_m},
                            )
        C2_XG_m.solve()
        values.append(np.maximum(
                C2_XG_m.solution.get_objective_value(),
                C_XG_m.solution.get_objective_value()
            ))
        ############################################
        ########## GENERALIZED X-POLYTOPE ##########
        ############################################
        # Pure integer
        # Add GX cuts  
        C_GX_p, best_GX_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_GX_p_'+str(row_ct))
        values.append(C_GX_p.solution.get_objective_value())
        # Add GXG cuts
        C_GXG_p, best_GXG_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_GXG_p_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_GXG_p = addUserCut(C, cont*0, LPS, "GX", 
                            cutDetails={'ans':best_GX_p},
                            )
        C2_GXG_p.solve()
        values.append(np.maximum(
                C2_GXG_p.solution.get_objective_value(),
                C_GXG_p.solution.get_objective_value()
            ))
        # Mixed integer
        # GX cut
        C_GX_m, best_GX_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_GX_m_'+str(row_ct))
        values.append(C_GX_m.solution.get_objective_value())
        # Add GXG cuts
        C_GXG_m, best_GXG_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose)
        names.append('C_GXG_m_'+str(row_ct))
        # Generating an GXG cuts with the best parameter of GX
        _, C2_GXG_m = addUserCut(C, cont, LPS, "GX", 
                            cutDetails={'ans':best_GX_m},
                            )
        C2_GXG_m.solve()
        values.append(np.maximum(
                C2_GXG_m.solution.get_objective_value(),
                C_GXG_m.solution.get_objective_value()
            ))
    return names, values


def getNumCut(M, filenames = False, verbose = False, solved = False):
    """
    ans = getNumCut(M):
    M : A CPLEX model or filename
    ans: A dictionary
    Given a model, presolves it aggressively, adds a bunch of cuts that and solves iteratively the LP relaxation and gives the details just before branching.
    ans["cuts"] = Number of cuts
    ans["LP"] = Final LP bound    
    """
    if filenames:
        Model = Cplex.cplex()
        Model.read(M)
    else:
        Model = M
    if not verbose:
        M.set_log_stream(None)                                          # Don't print log on screen
        M.set_results_stream(None)                                      # Don't print progress on screen    
    if not solved:
        Model.parameters.mip.limits.nodes.set(0)
        Model.solve()
    cuts = 0
    for i in Model.solution.MIP.cut_type:
        cuts = cuts + Model.solution.MIP.get_num_cuts(i)
    Model.parameters.mip.limits.nodes.set(9223372036800000000)
    ans = dict()
    ans["cuts"] = cuts
    ans["finalLP"] = Model.solution.MIP.get_best_objective()
    return ans   




def run_MIPLIB(problems = ['enlight9'], 
  rowlengths = [2,3], 
  nTrials = 2, 
  prefix = './MIPLIB/', 
  postfix = '.mps', 
  nCuts = 100,
  n_badrow = [1, 2],
  runGX = False,
  runX = False,
  verbose = False):
    """
    run_MIPLIB(problems = ['enlight9'], rowlengths = [2,3], nTrials = 2, prefix = './MIPLIB/', postfix = '.mps', nCuts=100, n_badrow = 1, runGX = False, runX = False, verbose=False)
    problems = set of MIPLIB problem names to run.
    prefix/postfix = Any relative path and extension for file names
    rowlengths: list containing number of row cuts to be iterated on
    nTrials: number of iterations on cut of each row length
    nCuts: number of Cuts
    n_badrow: number of "bad rows" to be pickd in each cut
    runGX: whether or not to run GX cuts
    runX: Whether or not to run X cuts
  
    Returns GXGvals/GXvals, both of shape
    (len(problems), len(rowlengths), nTrials)
    containing the objective value obtained in each.
    """
    Trials = range(nTrials)  
    LPvals = np.zeros((len(problems),))
    GMIvals = np.zeros((len(problems),))
    # Initializae GXGvals and GXvals if GX cuts are run
    if runGX:
        GXGvals = np.zeros((len(problems), len(rowlengths), len(n_badrow), len(Trials)))
        GXvals = np.zeros((len(problems), len(rowlengths), len(n_badrow), len(Trials)))
    # Initializae XGvals and Xvals if X cuts are run
    if runX:
        XGvals = np.zeros((len(problems), len(rowlengths), len(Trials)))
        Xvals = np.zeros((len(problems), len(rowlengths), len(Trials)))
    # Do this for each problem in under consderation
    for filename in problems:    
        # Reading the original MIPLIB problem
        C_org = cplex.Cplex()
        C_org.read(prefix+filename+postfix)
        int_var_org = np.array([0 if i=='C' else 1 for i in C_org.variables.get_types()])
        # Converting it into standard form
        C = Cplex2StdCplex(prefix+filename+postfix, MIP  = True, verbose = False)
        cont_var = np.array([1 if i=='C' else 0 for i in C.variables.get_types()])
        int_var = 1-cont_var
        C.set_problem_type(C.problem_type.LP)
        C.write(prefix+filename+'_std'+postfix)
        # Solving the LP relaxation of the standard form and getting solve information
        LPSolution = getfromCPLEX(C, verbose=False, ForceSolve=True, tableaux=False)
        print(LPSolution["Objective"])
        x_B = -LPSolution["Solution"][LPSolution["Basic"]]
        bad_rows = intRows(x_B,int_var[LPSolution["Basic"]].astype(int))
        print("ORIGINAL PROBLEM\n******************")
        print("nVar: "+str(C_org.variables.get_num())+ 
        "\n nCons: "+str(C_org.linear_constraints.get_num()) +
        "\n IntCon: "+str(np.sum(int_var_org)))
        print("\nSTANDARD PROBLEM\n*****************")
        print("nVar: "+str(C.variables.get_num())+ 
        "\n nCons: "+str(C.linear_constraints.get_num()) +
        "\n IntCon: "+ str (np.sum(int_var) ))
        print("OTHERS\n******")
        print("LP Objective: ", LPSolution["Objective"])
        print("# Integer constraints not satified in LP relaxation:", np.where(bad_rows)[0].shape[0])
        # Dealing with LP relaxation complete
        # Adding GMI cuts
        (A_GMI, b_GMI) = GMI(
                            LPSolution["Tableaux_NB"].todense().A, 
                            -LPSolution["Sol_Basic"], 
                            bad_rows, 
                            cont_var[LPSolution["NonBasic"]].astype(int)
                            )
        C_GMI = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                            NB = LPSolution["NonBasic"],
                            A_cut = A_GMI,
                            b_cut = b_GMI)
        GMIans = getfromCPLEX(C_GMI, tableaux = False, basic = False, TablNB = False)
        print('GMI:', GMIans["Objective"])
        # GMI complete
        # Adding Crosspolytope based cuts
        # Looping among all rowlengths required
        for nRows in rowlengths:
            # Looping over number of trials needed
            for Trial in Trials:
                # If GX cuts have to be done, then the following
                if runGX:
                    # In GX cuts, there is an option of choosing number of bad rows. Looping over all reqd values
                    for badrow_ct in n_badrow:
                        ans = Rows4Xcut(x_B, nRows, nCuts, int_var[LPSolution["Basic"]], badrow_ct)
                        # Calculating GX cuts
                        (A_GX, b_GX) = GXLift(LPSolution["Tableaux_NB"], 
                                            -LPSolution["Sol_Basic"],
                                            ans["RowMat"],
                                            ans["muMat"],
                                            ans["fMat"],
                                            cont_var[LPSolution["NonBasic"]].astype(int),
                                            sparse = True,
                                            verbose = verbose
                                            )
                        # creating GX model
                        C_GX = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                            NB = LPSolution["NonBasic"],
                                            A_cut = A_GX,
                                            b_cut = b_GX)
                        # creating GXG model
                        C_GXG = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                            NB = LPSolution["NonBasic"],
                                            A_cut = np.concatenate((A_GX , A_GMI),axis=0),
                                            b_cut = np.concatenate((b_GX,  b_GMI),axis=0))
                        # Solving the models with cuts
                        GXans = getfromCPLEX(C_GX, tableaux = False, basic = False, TablNB = False)
                        GXGans = getfromCPLEX(C_GXG, tableaux = False, basic = False, TablNB = False)
                        # Printing and storing the results
                        print(nRows,'row cut GX in Problem: ', filename, 'with badrow count: ', badrow_ct, '. Improvement: ', GXans["Objective"], GXGans["Objective"],sep = " ")
                        GXvals[problems.index(filename), rowlengths.index(nRows), n_badrow.index(badrow_ct), Trial] = GXans["Objective"]
                        GXGvals[problems.index(filename), rowlengths.index(nRows), n_badrow.index(badrow_ct), Trial] = GXGans["Objective"]
                # If X cuts have to be run
                if runX:
                    # Note that there is no looping over number of badrow selection. Number of badrow = number of rows here, necessarily.
                    ans = Rows4Xcut(x_B, nRows, nCuts, int_var[LPSolution["Basic"]], nRows)
                    # Calculating the X cuts
                    (A_X, b_X) = XLift(LPSolution["Tableaux_NB"], 
                                        -LPSolution["Sol_Basic"],
                                        ans["RowMat"],
                                        ans["muMat"],
                                        cont_var[LPSolution["NonBasic"]].astype(int),
                                        sparse = True,
                                        verbose = verbose
                                        )
                    # Creating the X model
                    C_X = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                    NB = LPSolution["NonBasic"],
                                    A_cut = A_X,
                                    b_cut = b_X)
                    # Creating the XG model
                    C_XG = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                        NB = LPSolution["NonBasic"],
                                        A_cut = np.concatenate((A_X , A_GMI),axis=0),
                                        b_cut = np.concatenate((b_X,  b_GMI),axis=0))
                    # Solving the models with cuts
                    Xans = getfromCPLEX(C_X, tableaux = False, basic = False, TablNB = False)
                    XGans = getfromCPLEX(C_XG, tableaux = False, basic = False, TablNB = False)
                    # Printing and storing the results
                    print(nRows,'row X cut in Problem: ', filename, Xans["Objective"], XGans["Objective"],sep = " ")
                    Xvals[problems.index(filename), rowlengths.index(nRows), Trial] = Xans["Objective"]
                    XGvals[problems.index(filename), rowlengths.index(nRows), Trial] = XGans["Objective"]
    # Returning appropriately based on inputs.
    if runX and runGX:
        return GXvals, GXGvals, Xvals, XGvals
    if runX:
        return Xvals, XGvals
    if runGX:
        return GXvals, GXGvals
    else:
        return GMIans

