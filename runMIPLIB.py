import numpy as np
import scipy as sp
import scipy.sparse

import scipy.sparse.linalg
import cplex

np.set_printoptions(precision=3)

from Cuts import *
from ReprVars import *
from CPLEXInterface import *

def run_compare_root(Batch = "int_A", Num_IP = 10, Nvar = 25, 
                Ncons = 10, NumRounds = 10, nRows = [2,3,5,10], 
                nCuts = 10, path = './', verbose = 0, scratch = './'):    
    values = []
    names = []
    problem = 0
    while problem < Num_IP:
        A = np.random.randint(-5,5,size = (Ncons, Nvar)) 
        # Generate a non-negative vector for variable values
        temp = np.ones((Nvar,1))
        # Choosing b this way ensures LP feasibility
        b = A.dot(temp)
        f = np.arange(Nvar)-np.round(Nvar/2)  
        cont = np.random.randint(0,2, size = (Nvar,))
        M = MIP(form = 1, data = {
            'Aeq':A,
            'beq':b,
            'f':f,
            'cont':cont
        })
        name = Batch + "_" + str(problem+1)
        _,v = compare_root_problem(M, NumRounds, name = name, nRows = nRows, nCuts = nCuts, verbose = verbose-1, scratch = scratch) 
        if v is None:
            continue
        M.write(name, path = path)
        np.savetxt(path + name + "_Sol.csv", np.array(v), delimiter = ',', fmt = '%6.6f')
        values.append(v)
        if verbose > 1:
            print(name +" completed")
        names.append(name)
        problem = problem + 1
    np.savetxt(path + Batch + "_Sol.csv", np.array(values), delimiter = ',', fmt = '%6.6f')
    np.savetxt(path + Batch + "_names.csv", np.array(names), delimiter = ',', fmt = '%s')
    return values


def CreateRandProb(Batch, name, path, Nvar, Ncons):
    A = np.random.randint(-5,5,size = (Ncons, Nvar)) 
    # Generate a non-negative vector for variable values
    temp = np.ones((Nvar,1))
    # Choosing b this way ensures LP feasibility
    b = A.dot(temp)
    f = np.arange(Nvar)-np.round(Nvar/2)  
    cont =np.random.randint(0,2, size = (Nvar,))
    M = MIP(form = 1, data = {
        'Aeq':A,
        'beq':b,
        'f':f,
        'cont':cont
    })    
    C = Py2Cplex(M)
    C.variables.set_types(
            [(i,"I") if j else (i,"C") for (i,j) in zip(range(Nvar),cont) ]
        )
    C.write(path + Batch + name + ".mps")


def run_Race_CPLEX_random(Batch = "A", NumIP = 100, Nvar = 50, Ncons =20, BestOf = 100, Roundlimit = 10, AbsRoundLim = 100, nCuts = 1, nBad = 1,nRows = 2 ,postfix = ".mps", scratch = "./", verbose = 0):
    """
    Runs NumIP number of simulations. Nvar/Ncons denote the problem sizes. 
    Uses Cplex to generate a bunch of rounds of cuts in root node. Does not go to branching. The objective obtaineed  here is 
    cplex_performance["finalLP"] and the number of cuts added by CPLEX is cplex_performance["cuts"]
    We add at most "RoundLimits" times the number of cuts that CPLEX had added. We see how many cuts do we require to beat CPLEX's performance.
    How do we add cuts?
        Loop (We add 1 cut to the LP relaxation. The 1 cut is chosen as the best cut out of "BestOf" cuts generated. Then the new LP is solved.)
    The above loop is done until we reach max allowed cuts or we beat CPLEX. Number of cuts needed to beat CPLEX is noted and returned
    """
    # Creating the problem
    values = []
    cplex_val = []
    print("**************************************")
    print("Number of problems to be run:", NumIP)
    print(str(Nvar)+" variables and "+str(Ncons)+" constraints in standard form.")
    print("Each cut chosen as best of " + str(BestOf)+ " GX cuts.")
    print("GX cuts are "+str(nRows)+" row cuts with "+str(nBad)+" bad rows.")
    print("**************************************")
    print("Simulation Starting ...")
    problem = 0
    while (problem < NumIP):
        f = np.arange(Nvar)-np.round(Nvar/2)
        A = np.random.randint(-5,5,(Ncons,Nvar))
        temp = np.ones((Nvar,1))
        b = A.dot(temp)
        cont = np.random.randint(0,2,(Nvar,))
        # Creating the MIP object
        M = MIP(form = 1, data = {
            'Aeq':A,
            'beq':b,
            'f':f,
            'cont':cont
        })
        name = Batch + "_" + str(problem+1)
        M.write(name = name, path =  scratch)        
        # Creating the object to add GX cuts:
        C_GX = Py2Cplex(M)
        if verbose <= 2:
            C_GX.set_log_stream(None)                                          # Don't print log on screen
            C_GX.set_results_stream(None)                                      # Don't print progress on screen    
            C_GX.set_warning_stream(None)
        C_GX.set_problem_name(name)        
        C_GX.solve()
        if C_GX.solution.get_status_string() != "optimal": # If LP optimal is not being obtained
            continue
        # Getting the solution in CPLEX
        C = Py2Cplex(M)
        t1 = [(i,"I") if j else (i,"C") for (i,j) in zip(range(Nvar),cont) ]
        C.variables.set_types(t1)
        cplex_performance = getNumCut(C, verbose = verbose - 2)
        cplex_val.append(cplex_performance["cuts"])
        if verbose > 1:
            print(cplex_performance)
        My_cont = M.cont.copy()
        value = float("inf")
        ActLim = min(
                    cplex_performance["cuts"]*Roundlimit,
                    AbsRoundLim
                    )
        for i in range(ActLim): 
            C_GX = ChooseBestCuts(C_GX, My_cont, cutDetails = {'nRows':nRows, 'nCuts':nCuts, 'nBad':nBad})
            My_cont = My_cont.tolist()
            My_cont.append(1)
            My_cont = np.array(My_cont)
            C_GX.solve()
            if verbose > 1:
                print("Round "+str(i+1), C_GX.solution.get_objective_value())
            if C_GX.solution.get_objective_value() >= cplex_performance["finalLP"]:
                value = i+1
                break
        values.append(value)            
        if verbose > 0:
            # print("Problem "+str(problem+1)+" completed with " + str(value) + " GX cuts to beat CPLEX, where CPLEX added " + str(cplex_performance["cuts"]) + " cuts")
            print(str(problem+1)+" " + str(value) + " " + str(cplex_performance["cuts"]))
        problem = problem + 1
    if verbose > 0:
        print(values,cplex_val,sep = "\n")
    return(values, cplex_val)


    






def compare_root_problem(M, NumRounds, nRows = [2,3,5,10], nBad = 1, nCuts = 10, verbose = 0, name = "_", isCplexObj = False, scratch = './'):
    """
    M is an "our" MIP object
    """
    if isCplexObj:
        C = M
        try:
            temp = C.variables.get_types()
            C.set_problem_type(C.problem_type.LP)
        except Exception as e:
            print("If CPLEX object is passed to compare_root_problem, it should be an MIP; "+str(e))
        cont = [1 if i == 'C' else 0 for i in temp]        
    else:        
        # Create Cplex Object
        C = Py2Cplex(M)
        cont = M.cont
    C.set_problem_name(name)
    values = []
    names = []
    # Solve the LP relaxation
    LPS = getfromCPLEX(C, ForceSolve=True)  
    if C.solution.get_status_string() != "optimal": # If LP optimal is not being obtained
        if verbose > 0:
            print (name + ' '+C.solution.get_status_string())
        return None, None  
    bad = np.where(intRows(LPS["Sol_Basic"], 1-cont[LPS["Basic"]].astype(int)))[0].size
    if bad < np.max(np.array(nRows)):
        if verbose > 0:
            print(name + ' ' + "Not enough bad rows. # Bad rows: " + str(bad))
        return None, None
    names.append('LP')
    values.append(LPS["Objective"])
    # Add GMI cuts for pure integer version
    C_GMI_p,_ = addUserCut(C, cont*0, LPS, verbose = verbose - 1, scratch = scratch)
    C_GMI_p.solve()
    names.append('GMI_p')
    values.append(C_GMI_p.solution.get_objective_value())
    # Add GMI cuts for mixed integer version
    C_GMI_m,_ = addUserCut(C, cont, LPS, verbose = verbose - 1, scratch = scratch)
    C_GMI_m.solve()
    names.append('GMI_m')
    values.append(C_GMI_m.solution.get_objective_value())
    if verbose > 0:
        print("GMIs generated")
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
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch )
        names.append('C_X_p_'+str(row_ct))
        values.append(C_X_p.solution.get_objective_value())
        # Add XG cuts
        C_XG_p, best_XG_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_XG_p_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_XG_p = addUserCut(C, cont*0, LPS, "X", 
                            cutDetails={'ans':best_X_p}, scratch = scratch
                            )
        C2_XG_p.solve()
        values.append(np.maximum(
                C2_XG_p.solution.get_objective_value(),
                C_XG_p.solution.get_objective_value()
            ))
        if verbose > 0:
            print(str(row_ct)+" row X cuts generated for pure")        
        # Mixed integer
        # X cut
        C_X_m, best_X_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_X_m_'+str(row_ct))
        values.append(C_X_m.solution.get_objective_value())
        # Add XG cuts
        C_XG_m, best_XG_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "X",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_XG_m_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_XG_m = addUserCut(C, cont, LPS, "X", 
                            cutDetails={'ans':best_X_m}, scratch = scratch
                            )
        C2_XG_m.solve()
        values.append(np.maximum(
                C2_XG_m.solution.get_objective_value(),
                C_XG_m.solution.get_objective_value()
            ))
        if verbose > 0:
            print(str(row_ct)+" row X cuts generated for mixed")        
        ############################################
        ########## GENERALIZED X-POLYTOPE ##########
        ############################################
        # Pure integer
        # Add GX cuts  
        C_GX_p, best_GX_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_GX_p_'+str(row_ct))
        values.append(C_GX_p.solution.get_objective_value())
        # Add GXG cuts
        C_GXG_p, best_GXG_p = ChooseBestCuts(C, cont*0, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_GXG_p_'+str(row_ct))
        # Generating an XG cuts with the best parameter of X
        _, C2_GXG_p = addUserCut(C, cont*0, LPS, "GX", 
                            cutDetails={'ans':best_GX_p}, scratch = scratch
                            )
        C2_GXG_p.solve()
        values.append(np.maximum(
                C2_GXG_p.solution.get_objective_value(),
                C_GXG_p.solution.get_objective_value()
            ))
        if verbose > 0:
            print(str(row_ct)+" row GX cuts generated for pure")        
        # Mixed integer
        # GX cut
        C_GX_m, best_GX_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds, 
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_GX_m_'+str(row_ct))
        values.append(C_GX_m.solution.get_objective_value())
        # Add GXG cuts
        C_GXG_m, best_GXG_m = ChooseBestCuts(C, cont, getfromCPLEX_Obj=LPS, 
                    cutType = "GX",
                    Nrounds = NumRounds - 1,    # -1 because the best parameters from X will also be used here
                    withGMI = True,             # Differene between X and XG is by controlling this
                    cutDetails={'nRows':row_ct, 'nCuts':nCuts, 'nBad':1}, 
                    return_bestcut_param=True, verbose = verbose - 1, scratch = scratch)
        names.append('C_GXG_m_'+str(row_ct))
        # Generating an GXG cuts with the best parameter of GX
        _, C2_GXG_m = addUserCut(C, cont, LPS, "GX", 
                            cutDetails={'ans':best_GX_m}, scratch = scratch
                            )
        C2_GXG_m.solve()
        values.append(np.maximum(
                C2_GXG_m.solution.get_objective_value(),
                C_GXG_m.solution.get_objective_value()
            ))
        if verbose > 0:
            print(str(row_ct)+" row GX cuts generated for mixed")        
    return names, values


def getNumCut(M, filenames = False, verbose = 0, solved = False, returnModel = False):
    """
    ans = getNumCut(M):
    M : A CPLEX model or filename
    ans: A dictionary
    Given a model, presolves it aggressively, adds a bunch of cuts that and solves iteratively the LP relaxation and gives the details just before branching.
    ans["cuts"] = Number of cuts
    ans["LP"] = Final LP bound    
    """
    if filenames:
        Model = cplex.Cplex()
        Model.read(M)
    else:
        Model = M
    if verbose <= 0:
        Model.set_log_stream(None)                                          # Don't print log on screen
        Model.set_results_stream(None)                                      # Don't print progress on screen    
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
    if returnModel:
        return ans, Model
    else:
        return ans   


def run_MIPLIB_root(problemlist,
        rowlengths = [2,5,10], 
        nTrials = 10, 
        prefix = './MIPLIB/', 
        postfix = '.mps', 
        nCuts = 100,
        n_badrow = [1, 5],
        runGX = False,
        runX = False,
        verbose = 0,
        scratch = './',
        saveDict = True
    ):
    allFile = []
    t2 = open(problemlist,"r")
    t3 = t2.readlines()
    for i in t3:
        allFile.append(i[0:i.find('.')])
    t2.close()
    if verbose > 0:
        print("File Set: ", allFile)   
        print("nTrials: "+str(nTrials), "nCuts: "+str(nCuts), "n_badrow: "+str(n_badrow), sep = "\n")
        print("*********************")
        print("*****Run Started*****")
        print("*********************")      
    t1 = run_MIPLIB(allFiles, rowlengths, nTrials, prefix, postfix, nCuts, n_badrow, runGX, runX, verbose - 1, scratch, saveDict)
    if verbose > 0:
        print("**********************")
        print("*****Run Complete*****")
        print("**********************")
        print(t1)




# This is required to retrieve data from a saved dictionary, just in case 
nan = np.nan

def run_MIPLIB(problems = ['enlight9'], 
  rowlengths = [2,3], 
  nTrials = 2, 
  prefix = './MIPLIB/', 
  postfix = '.mps', 
  nCuts = 100,
  n_badrow = [1, 2],
  runGX = False,
  runX = False,
  verbose = 0,
  scratch = './',
  saveDict = False):
    """
    run_MIPLIB(problems = ['enlight9'], rowlengths = [2,3], nTrials = 2, prefix = './MIPLIB/', postfix = '.mps', nCuts=100, n_badrow = 1, runGX = False, runX = False, verbose=0)
    problems = set of MIPLIB problem names to run.
    prefix/postfix = Any relative path and extension for file names
    rowlengths: list containing number of row cuts to be iterated on
    nTrials: number of iterations on cut of each row length
    nCuts: number of Cuts
    n_badrow: number of "bad rows" to be picked in each cut
    runGX: whether or not to run GX cuts
    runX: Whether or not to run X cuts
    saveDict: SHould we save a dictionary with the results?
  
    Returns GXGvals/GXvals, both of shape
    (len(problems), len(rowlengths), nTrials)
    containing the objective value obtained in each.
    """
    Trials = range(nTrials)  
    LPvals = np.zeros((len(problems),))
    GMIvals = np.zeros((len(problems),))
    AllCutSol = dict()    
    # Do this for each problem in under consideration
    for filename in problems:    
        if verbose > 0:
            print("Running: "+str(filename))
        cutValues = dict()
        # Reading the original MIPLIB problem
        C_org = cplex.Cplex()
        C_org.read(prefix+filename+postfix)
        int_var_org = np.array([0 if i=='C' else 1 for i in C_org.variables.get_types()])
        # Converting it into standard form
        C = Cplex2StdCplex(prefix+filename+postfix, MIP  = True, verbose = verbose-1)
        cont_var = np.array([1 if i=='C' else 0 for i in C.variables.get_types()])
        int_var = 1-cont_var
        C.set_problem_type(C.problem_type.LP)
        C.write(prefix+filename+'_std'+postfix)
        # Solving the LP relaxation of the standard form and getting solve information
        LPSolution = getfromCPLEX(C, verbose=verbose-1, ForceSolve=True, tableaux=False)
        x_B = -LPSolution["Solution"][LPSolution["Basic"]]
        bad_rows = intRows(x_B,int_var[LPSolution["Basic"]].astype(int))
        if verbose > 1:
            print(LPSolution["Objective"])
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
        cutValues["LP"] = LPSolution["Objective"]
        cutValues["badrow"] = np.where(bad_rows)[0].shape[0]
        # Dealing with LP relaxation complete
        # Adding GMI cuts
        (A_GMI, b_GMI) = GMI(
                            LPSolution["Tableaux_NB"].todense().A, 
                            LPSolution["Sol_Basic"], 
                            bad_rows, 
                            cont_var[LPSolution["NonBasic"]].astype(int)
                            )
        C_GMI = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                            NB = LPSolution["NonBasic"],
                            A_cut = A_GMI,
                            b_cut = b_GMI, scratch = scratch)
        GMIans = getfromCPLEX(C_GMI, tableaux = False, basic = False, TablNB = False)
        if verbose > 1:
            print('GMI:', GMIans["Objective"])
        cutValues["GMI"] = GMIans["Objective"]
        # GMI complete
        # Adding Crosspolytope based cuts
        # Looping among all rowlengths required
        for nRows in rowlengths:
            # Initialize GXGvals and GXvals if GX cuts are run            
            if runGX:
                GXGvals = np.zeros((len(n_badrow), len(Trials)))
                GXvals = np.zeros((len(n_badrow), len(Trials)))
            # Initialize XGvals and Xvals if X cuts are run
            if runX:
                XGvals = np.zeros((len(Trials),))
                Xvals = np.zeros((len(Trials),))
            # Looping over number of trials needed
            for Trial in Trials:
                # If GX cuts have to be done, then the following
                if runGX:
                    # In GX cuts, there is an option of choosing number of bad rows. Looping over all reqd values
                    for badrow_ct in n_badrow:
                        ans = Rows4Xcut(x_B, nRows, nCuts, int_var[LPSolution["Basic"]], badrow_ct)
                        if ans is None: # Problem occurred in X cut parameter generation. This can happen if there are insufficient badrows
                            print(nRows,'row GX cut in Problem: ', filename, "not possible", sep = " ")
                            GXvals[n_badrow.index(badrow_ct), Trial] = None
                            GXGvals[n_badrow.index(badrow_ct), Trial] = None
                        else:
                            # Calculating GX cuts
                            (A_GX, b_GX) = GXLift(LPSolution["Tableaux_NB"], 
                                                -LPSolution["Sol_Basic"],
                                                ans["RowMat"],
                                                ans["muMat"],
                                                ans["fMat"],
                                                cont_var[LPSolution["NonBasic"]].astype(int),
                                                sparse = True,
                                                verbose = verbose-1
                                                )
                            # creating GX model
                            C_GX = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                                NB = LPSolution["NonBasic"],
                                                A_cut = A_GX,
                                                b_cut = b_GX, scratch = scratch)
                            # creating GXG model
                            C_GXG = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                                NB = LPSolution["NonBasic"],
                                                A_cut = np.concatenate((A_GX , A_GMI),axis=0),
                                                b_cut = np.concatenate((b_GX,  b_GMI),axis=0), scratch = scratch)
                            # Solving the models with cuts
                            GXans = getfromCPLEX(C_GX, tableaux = False, basic = False, TablNB = False)
                            GXGans = getfromCPLEX(C_GXG, tableaux = False, basic = False, TablNB = False)
                            # Printing and storing the results
                            if verbose > 1:
                                print(nRows,'row cut GX in Problem: ', filename, 'with badrow count: ', badrow_ct, '. Improvement: ', GXans["Objective"], GXGans["Objective"],sep = " ")
                            GXvals[n_badrow.index(badrow_ct), Trial] = GXans["Objective"]
                            GXGvals[n_badrow.index(badrow_ct), Trial] = GXGans["Objective"]
                # If X cuts have to be run
                if runX:
                    # Note that there is no looping over number of badrow selection. Number of badrow = number of rows here, necessarily.
                    ans = Rows4Xcut(x_B, nRows, nCuts, int_var[LPSolution["Basic"]], nRows)
                    if ans is None: # Problem occurred in X cut parameter generation. This can happen if there are insufficient badrows
                        print(nRows,'row X cut in Problem: ', filename, "not possible", sep = " ")
                        Xvals[Trial] = None
                        XGvals[Trial] = None
                    else:
                        # Calculating the X cuts
                        (A_X, b_X) = XLift(LPSolution["Tableaux_NB"], 
                                            -LPSolution["Sol_Basic"],
                                            ans["RowMat"],
                                            ans["muMat"],
                                            cont_var[LPSolution["NonBasic"]].astype(int),
                                            sparse = True,
                                            verbose = verbose-1
                                            )
                        # Creating the X model
                        C_X = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                        NB = LPSolution["NonBasic"],
                                        A_cut = A_X,
                                        b_cut = b_X, scratch = scratch)
                        # Creating the XG model
                        C_XG = addCuts2Cplex(filename = prefix+filename+'_std'+postfix,
                                            NB = LPSolution["NonBasic"],
                                            A_cut = np.concatenate((A_X , A_GMI),axis=0),
                                            b_cut = np.concatenate((b_X,  b_GMI),axis=0), scratch = scratch)
                        # Solving the models with cuts
                        Xans = getfromCPLEX(C_X, tableaux = False, basic = False, TablNB = False)
                        XGans = getfromCPLEX(C_XG, tableaux = False, basic = False, TablNB = False)
                        # Printing and storing the results
                        if verbose > 1:
                            print(nRows,'row X cut in Problem: ', filename, Xans["Objective"], XGans["Objective"],sep = " ")
                        Xvals[Trial] = Xans["Objective"]
                        XGvals[Trial] = XGans["Objective"]
            if runGX or runX:
                cutValues[str(nRows)] = dict()
            if runGX:                
                cutValues[str(nRows)]["GX"] = GXvals.tolist()
                cutValues[str(nRows)]["GXG"] = GXGvals.tolist()
            if runX:
                cutValues[str(nRows)]["X"] = Xvals.tolist()
                cutValues[str(nRows)]["XG"] = XGvals.tolist()
        AllCutSol[filename] = cutValues
        if verbose > 0:
            print(AllCutSol)
    # Returning appropriately based on inputs.
    if runX or runGX:
        if saveDict:
            myFile = open( scratch +str(len(problems))+"_prob_"+str(nCuts)+"_cuts_"+str(nTrials)+"_trials_"+str(int(C.get_time()*1000)) + ".txt" , "w")
            myFile.write(str(AllCutSol))
            myFile.close()
        return AllCutSol
    else:
        return GMIans

