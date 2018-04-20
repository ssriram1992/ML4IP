import numpy as np
import scipy as sp
import scipy.sparse                 # To handle sparse data
import scipy.sparse.linalg          # To perform linear algrbra operation on sparse data
import cplex
import copy
import io as  cStringIO
import re
from scipy.stats import variation
import math

# Making changes in Sri branch
class MIP:
    """
    A class to represent Mixed-integer-programs in standard form
    Standard form is the follows
    \min f^Tx subject to
    Aeq x = beq
    x >= 0
    x \in \Z for x\ not\in cont

    MIP.f, MIP.Aeq, MIP.beq, MIP.cont are used to save the objects
    Important function for feature extraction:
    MIP.features(tol = 1e-9)
        Extracts all features from an instance and returns a dictionary with the feature names and
        feature values.

    Flags:
    # These are binary falgs which are switched when a computation heavy operation is done
    # so that the operation need not be repeated unnecessarily.
    # Do not alter them if unsure.
    CplexMade
        True if corresponding CPLEX Object is made
    LPSolved
    GraphDrawn
    TableGraphDrawn
    Probed

    Other objects saved:
    MIP.CplexObjectLP
        Created after MakeCplex function. The Cplex Object which stores the same MIP as LP
    self.CplexObjectMIP
        Created after MakeCplex function. The Cplex Object which stores the same MIP as MIP
    MIP.LPInfo
        Created after LPSolve function. Stores
            LPSolution vector
            LPSolution Objective
            LPSolution simplex tableau
            LPSolution basic variable set and non Basic variable set
            LPSolution NonBasic Tableaux
    MIP.Probingdict
        Created after getProbingFeatures function. This is a dictionary object and stores the
        probing result. This is required as multiple probing can give different results.

    Methods of use:
    MIP.write(name, path="./")
        Writes 4 csv files in the given path with the name prefix given, saving Aeq, beq, obj and cont
    MIP.MakeCplex()
        Makes a Cplex object of original MIP as well as the LP relaxation
        of the given MIP instance and saves it.
    MIP.SolveLP()
        Solve the LP relaxation of the problem and create the LPInfo object

    Features implemented:
    MIP.size()
        Returns a dictionary with {'nVar', 'nCons', 'nInt'} containing number of variables,
        constraints and number of integer variables respectively.
    MIP.VCGraph(tol = 1e-9)
        Returns a dictionary with {'VCG_V_mean, 'VCG_V_std', 'VCG_V_min', 'VCG_V_max', 'VCG_V_25p', 'VCG_V_75p',
                                   'VCG_C_mean, 'VCG_C_std', 'VCG_C_min', 'VCG_C_max', 'VCG_C_max', 'VCG_C_25p'}
        VCG referes to variable constraint graph
        _V_ refers to variable node statistics
        _C_ refers to constraint node statistics
    MIP.LPObjVal()
        Returns LP objective value as dictionary {'LPObjective'}
    MIP.LPIntegerSlack()
        Returns a dictionary with {'Slack_1_mean', 'Slack_1_std', 'Slack_1_min', 'Slack_1_max', 'Slack_1_25p', 'Slack_1_75p', 'Slack_1_norm',
                                   'Slack_2_mean', 'Slack_2_std', 'Slack_2_min', 'Slack_2_max', 'Slack_2_25p', 'Slack_2_75p', 'Slack_2_norm',
                                   "Slack_1_nonzero"}
        Integer slack vectors are calculated as follows
        Integer slack vector 1 is a vector of size equal to the number of integer variables in the problem
        For each integer variable x_i, this contains the number x_i - np.floor(x_i).
        Integer slack vector 2 is a vector of size equal to the number of variables in the problem
        If x_i is a continuous variable, then the i-th coordinate of this vector is 0
        else the i-th coordinate is  x_i - np.floor(x_i)
    MIP.VCGraph(tol = 1e-9)
        Returns a dictionary with
        {'VG_V_mean', 'VG_V_std', 'VG_V_min', 'VG_V_max', 'VG_V_25p', 'VG_V_75p', 'EdgeDens'}
    MIP.stdObjM()
        Returns the standard deviation of normalized coefficients as dictionary {'stdObjM'}
    MIP.stdObjN()
        Returns a dictionary with {'stdObjN', 'stdObjRootN'}
    MIP.AeqNormStats()
        Returns a dictionary with {'AeqNormMean', 'AeqNormStd'}
    MIP.CVStats()
        Returns a dictionary with {'CVMean', 'CVStd'}
    MIP.AOverb()
        Returns a dictionary with {'MinPos', 'MaxPos', 'MinNeg', 'MaxNeg'}
    MIP.OnetoAllA()
        Returns a dictionary with {'MinPosPos', 'MaxPosPos', 'MinPosNeg', 'MaxPosNeg', 'MinNegPos', 'MaxNegPos',
                                'MinNegNeg', 'MaxNegNeg'}
    MIP.getProbingFeatures(TL=10)
        Inputs time limit in seconds for CPLEX probing of the MIP.
        Returns a dictionary with {'numRowsPresolved', 'numColsPresolved', 'numNonzerosPresolved',
                                    'totalPresolveTime', 'totalProbingTime', 'cliqueTable',
                                    'numCuts' (of each type),
                                    'numCutsTotal', 'numIter', 'numNodesProc'}
    MIP.geometricFeatures()
        Returns a dictionary with 
    """
    def __init__(self, form = 0, data = dict(),filenames = False, delimiter = ','):
        """
        form        : = 0 implies empty initialization
                      = 1 implies intialilzation in standard form. (f, Aeq, beq, cont are given. All x >=0 taken automatically)
                      = 2 implies explicit definition of inequalities, equalities, lower bounds, upper bounds, integrality constraints
        data        : Dictionary containing everything that is required to initialize
        filenames   : Whether the values in the data dictionary are filenames or variable names.
        delimiter   : If values are read from a csvfile, what is the delimiter character?
        """
        self.CplexMade = False                    # Boolean to store if Cplex object is made
        self.LPSolved = False                     # Boolean to store if LP relaxation is solved
        self.GraphDrawn = False                   # Boolean to store if the Variable Constraint Graph is drawn
        self.TableGraphDrawn = False              # Boolean to store if the Variable Constraint Graph is drawn for simplex tableaux
        self.Probed = False                       # Boolean to store if probing is done
        if 'name' in data:                        # If the problem has to be named
            self.name = data["name"]
        else:
            self.name = ""
        if form == 0:
            # Blank initialization
            self.f = np.array([[]]).reshape((0,1)) # Minimization objjective
            self.Aeq = np.array([[]]).reshape((0,0)) # Equality constraint matrix
            self.beq = np.array([[]]).reshape((0,1)) # Equality constraint RHS
            self.cont = np.array([])               # 1/0 vector indicating if variable is continuous
        if form == 1:
            # Initialization in standard form
            f = data["f"]
            Aeq = data["Aeq"]
            beq = data["beq"]
            cont = data["cont"]
            if filenames:
                self.f = np.genfromtxt(f, delimiter=delimiter)
                self.Aeq = np.genfromtxt(Aeq, delimiter=delimiter)
                self.beq = np.genfromtxt(beq, delimiter=delimiter)
                self.cont = np.genfromtxt(cont, delimiter=delimiter)
            else:
                self.f = f.copy()
                self.Aeq = Aeq.copy()
                self.beq = beq.copy()
                self.cont = cont.copy()
        if form == 2:
            # Initialization in general form. To be converted to standard form and stored.
            # This can be used to convert problems in the form
            # min c^T x subject to
            # A x <= b
            # Aeq x = beq
            # lb <= x <= ub
            # x_i \in \Z if i \in intcon
            f = data["f"]
            Aeq = data["Aeq"]
            beq = data["beq"]
            cont = data["cont"]
            A = data["A"]
            b = data["b"]
            lb = data["lb"]
            ub = data["ub"]
            if filenames:
                f2 = np.genfromtxt(f, delimiter=delimiter)
                A2 = np.genfromtxt(A, delimiter=delimiter)
                b2 = np.genfromtxt(b, delimiter=delimiter)
                Aeq2 = np.genfromtxt(Aeq, delimiter=delimiter)
                beq2 = np.genfromtxt(beq, delimiter=delimiter)
                lb2  = np.genfromtxt(lb, delimiter=delimiter)
                ub2 = np.genfromtxt(ub, delimiter=delimiter)
                cont2 = np.genfromtxt(cont, delimiter=delimiter)
            else:
                f2 = f.copy().reshape((np.size(f),1))
                Aeq2 = Aeq.copy()
                beq2 = beq.copy().reshape((np.size(beq),1))
                cont2 = cont.copy().reshape((np.size(cont),1))
                A2 = A.copy()
                b2 = b.copy().reshape((np.size(b),1))
                lb2 = lb.copy().reshape((np.size(f),1))
                ub2 = ub.copy().reshape((np.size(f),1))
            nvar = np.size(f2)
            #
            lb_count = 0
            ub_count = 0
            # Defining Alb, Aub, blb and bub to convert bound constraints to standard form
            Aub = np.zeros((np.sum(ub2<np.inf),nvar))
            temp = np.sum(np.all((lb2!=0, lb2>-np.inf),axis=0)) #Number of lower bound constraints
            # LB constraints come, only if lb is neither 0 nor -Infinity
            Alb = np.zeros((temp,nvar))
            bub = np.zeros((np.sum(ub2<np.inf),1))
            blb = np.zeros((temp,1))
            if A2.size == 0:
                A2 = A2.reshape(0, nvar)
            if Aeq2.size == 0:
                Aeq2 = Aeq2.reshape(0, nvar)
            for i in np.arange(nvar):
                # Converting bound constraints to <= constraints
                # Ensure all variables satisfy >= 0
                if ub2[i, 0] < np.inf:
                    # If upper bound is finite, then a constraint has to be added in standard form
                    Aub[ub_count, i] = 1
                    bub[ub_count, 0] = ub2[i]
                    ub_count = ub_count + 1
                if lb2[i,0] < 0:
                    # If lower bound is less than 0, a constraint has to be added
                    # In fact a new variable has to be added. So the sizes of all the
                    # other entities have to be changed.
                    # i.e., rewriting x = x1 + x2 with x1 >=0 ,x2 >= 0
                    f2 = np.vstack((f2, [[-f2[i,0]]])) #Corresponding "-c_i" for new term in obj
                    cont2 = np.vstack((cont2, [[cont2[i,0]]]))
                    t1 = A2.shape[0] # Number of inequality constraints
                    if A2.size == 0:
                        t1 = 0
                    t2 = Aeq2.shape[0] # Number of equality constraints
                    if Aeq2.size == 0:
                        t2 = 0
                    t3 = Aub.shape[0] # Number of upperbound constraints
                    t4 = Alb.shape[0] # Number of lowerbound constraints
                    # Adding a column to constraint matrices with sign flipped, for the new variable
                    A2 = np.hstack((A2,
                                    -A2[:,i].reshape(t1,1)
                                    ))
                    Aeq2 = np.hstack((Aeq2,
                                    -Aeq2[:,i].reshape(t2,1)
                                    ))
                    Aub = np.hstack((Aub,
                                -Aub[:,i].reshape(t3,1)
                                ))
                    Alb = np.hstack((Alb,
                                -Alb[:,i].reshape(t4,1)
                                ))
                    if lb2[i,0]>-np.inf:
                        # However if lower bound is not -Inf (i.e., finite), then that constraint is
                        # added
                        Alb[lb_count, i] = -1
                        Alb[lb_count, np.size(f2)-1] = 1 #last variable so far
                        blb[lb_count, 0] = -lb2[i,0]
                        lb_count = lb_count+1
                else:
                    if lb2[i,0] > 0:
                        # If lower bound is greater than 0, constraint has to be added
                        Alb[lb_count,i] = -1
                        blb[lb_count, 0] = -lb2[i,0]
                        lb_count = lb_count+1
            # Count the number of constraints
            nineq = np.size(b2) # Number of straight forward inequalities
            neq = np.size(beq2) # Number of straight forward equalities
            nlb = np.size(blb) # Number of lower bound inequalities
            nub = np.size(bub) # Number of upper bound inequalities
            # Now convert inequality constraints to equality constraints
            # by adding slack variables (identity matrices to ineq constraints)
            A2 = np.hstack((A2, np.identity(nineq), np.zeros((nineq, nlb+nub))))
            Alb = np.hstack((Alb, np.zeros((nlb,nineq)),np.identity(nlb),np.zeros((nlb,nub))))
            Aub = np.hstack((Aub, np.zeros((nub,nineq+nlb)),np.identity(nub)))
            Aeq2 = np.hstack((Aeq2, np.zeros((neq, nineq+nlb+nub))))
            # Giving "0" in the objectives for slack variables
            f2 = np.vstack((f2, np.zeros((nineq+nlb+nub,1))))
            # Slack variables are assumed to be continuous
            # All input matrices are integers, this can be strengthened and slacks can be called
            # as integer variables. Perhaps that can give stronger cuts!
            cont2 = np.vstack((cont2, np.ones((nineq+nlb+nub,1))))
            # Creating the final A and b matrices for the standard form
            self.Aeq = np.vstack((Aeq2, Alb, Aub, A2))
            self.beq = np.vstack((beq2, blb, bub, b2))
            self.f = f2
            self.cont = cont2
            # Ensure b in standard form is non-negative
            negb = np.where(self.beq<0)[0] # Collect those rows and invert sign of those rows
            self.Aeq[negb,:] = -self.Aeq[negb,:]
            self.beq[negb,:] = -self.beq[negb,:]
        # End of function
    #######################
    # General purpose functions
    def write(self, name, path ='./'):
        np.savetxt(path + name + '_Aeq.csv', self.Aeq, delimiter = ',')
        np.savetxt(path + name + '_beq.csv', self.beq, delimiter = ',')
        np.savetxt(path + name + '_obj.csv', self.f, delimiter = ',')
        np.savetxt(path + name + '_cont.csv', self.cont, delimiter = ',')
    def MakeCplex(self):
        """
        Makes a Cplex object of the given MIP instance and saves it as MIP.CplexObjectLP
        """
        if not self.CplexMade:
            C = Py2Cplex(self)
            self.CplexObjectLP = C
            C2 = Py2CplexMIP(self)
            self.CplexObjectMIP = C2
            self.CplexMade = True
    def LPSolve(self):
        """
        Function, that uses the information in f, Aeq, beq and solves the LP relaxation of the problem.
        The function should return LP solution vector, LP objective value,
        and the set of optimal Basic Variables/Non-basic variables (so that we
        can calculate the tableaux externally). Optionally, the tableaux can be
        directly returned from this function.
        """
        if not self.LPSolved: # If LP is already solved - then nothing to do! Don't repeat the operation again!
            self.MakeCplex() # Make the CPLEX Object!
            self.LPInfo = getfromCPLEX(self.CplexObjectLP,
                                            solution = True,
                                            objective = True,
                                            tableaux = False,
                                            basic = False,
                                            TablNB = True, # Needed for geometric features
                                            precission = 13
                                        )
            self.LPSolved = True
    # End of general methods
    #######################
    # Feature extraction
    def features(self, tol = 1e-9, returnAsVect = False, returnNames = False):
        """
        Combines all the extracted features and returns one big feature dictionary
        """
        feature = {}
        self.MakeCplex()
        self.LPSolve()
        feature.update(self.size()) # Adding size related features
        feature.update(self.VCGraph(tol)) # Adding Variable-Constraint Graph features
        feature.update(self.VGraph(tol))  # Adding variable graph features
        feature.update(self.LPIntegerSlack()) # LP integer slack features
        feature.update(self.LPObjVal()) # LP objective value
        feature.update(self.getProbingFeatures()) # MIP probing features
        feature.update(self.stdObjM()) # Standard deviation of normalized coefficients: f_i/Number_of_constraints
        feature.update(self.stdObjN()) # Standard deviation of ci/ni
        feature.update(self.AeqNormStats()) # Stats of Aij/bi
        feature.update(self.CVStats()) # Stats of abs non zero entries each row
        feature.update(self.AOverb()) # Min/Max for ratios of constraint coeffs
        feature.update(self.OnetoAllA()) # Min/max for one-to-all coeff ratios
        feature.update(self.geometricFeatures()) # Geometry of the columns of Aeq
        if returnAsVect:
            K = list(feature.keys())
            K.sort()
            feature_vector =  [self.name] + [feature[i] for i in K]
            K = ["Name"] + K
            if returnNames:
                return feature_vector, K
            else:
                return feature_vector
        else:
            feature["Name"] = self.name
            return feature
    def size(self):
        """
        Returns the number of variables, number of constraints and number of integer
        variables in the IP
        """
        nVar = np.size(self.f)
        nCons = np.size(self.beq)
        nInt = nVar - np.sum(self.cont)
        return {'nVar':nVar, 'nCons':nCons, 'nInt':nInt, 'PercentageInteger':1.0*nInt/nVar}
    def VCGraph(self, tol = 1e-9):
        """
        Returns statistics of the variable constraint graph.
        Variable constraint graph is a bipartite graph with the set of variables as one independent set and
        the set of constraints as another independednt set. An edge between a variable and constraint is
        said to exist if and only if the said variable appears in the said constraint with non-zero
        coefficient (theoretically). In this implementation any number with absolute value less than tol
        is considered as zero.
        This function returns a dictionary with some statistics on the graph.
        """
        VCG = (abs(self.Aeq) < tol)*1 # Binary Matrix of same dimension as Aeq.
        Variable_Node_vec = np.sum(VCG, axis = 0)   # Node degree of each variable node
        Constraint_Node_vec = np.sum(VCG, axis = 1) # Node degree of each constraint node
        return {
            'VCG_V_mean':   np.mean(Variable_Node_vec),              # Mean degree of variable node
            'VCG_V_std' :   np.std(Variable_Node_vec),               # Std. dev of degree of variable node
            'VCG_V_min' :   np.min(Variable_Node_vec),               # Min degree of a variable node
            'VCG_V_max' :   np.max(Variable_Node_vec),               # Max degree of a variable node
            'VCG_V_25p' :   np.percentile(Variable_Node_vec, 25),    # 25th percentile degree of a variable node
            'VCG_V_75p' :   np.percentile(Variable_Node_vec, 75),    # 75th percentile degree of a variable node
            'VCG_C_mean':   np.mean(Constraint_Node_vec),            # Mean degree of constraint node
            'VCG_C_std' :   np.std(Constraint_Node_vec),             # Std. dev of degree of constraint node
            'VCG_C_min' :   np.min(Constraint_Node_vec),             # Min degree of a constraint node
            'VCG_C_max' :   np.max(Constraint_Node_vec),             # Max degree of a constraint node
            'VCG_C_25p' :   np.percentile(Constraint_Node_vec, 25),  # 25th percentile degree of a constraint node
            'VCG_C_75p' :   np.percentile(Constraint_Node_vec, 75)   # 75th percentile degree of a constraint node
        }
    def LPObjVal(self):
        """
        Returns the objective Value of LP relaxation
        """
        self.LPSolve()
        return {'LPObjective':self.LPInfo["Objective"]}
    def LPIntegerSlack(self, tol = 1e-9):
        """
        Returns statistics of the integer slack vector 1 and integer slack vector 2
        Integer slack vectors are calculated as follows
        Integer slack vector 1 is a vector of size equal to the number of integer variables in the problem
        For each integer variable x_i, this contains the number x_i - np.floor(x_i).
        Integer slack vector 2 is a vector of size equal to the number of variables in the problem
        If x_i is a continuous variable, then the i-th coordinate of this vector is 0
        else the i-th coordinate is  x_i - np.floor(x_i)
        """
        self.LPSolve()
        LPSol = self.LPInfo["Solution"].squeeze()
        slack2 = (LPSol - np.floor(LPSol))*(1-self.cont)
        slack1 = slack2[self.cont==0]
        return {
            'Slack_1_mean'   :   np.mean(slack1),
            'Slack_1_std'    :   np.std(slack1),
            'Slack_1_min'    :   np.min(slack1),
            'Slack_1_max'    :   np.max(slack1),
            'Slack_1_25p'    :   np.percentile(slack1, 25),
            'Slack_1_75p'    :   np.percentile(slack1, 75),
            'Slack_1_norm'   :   np.linalg.norm(slack1),
            'Slack_2_mean'   :   np.mean(slack2),
            'Slack_2_std'    :   np.std(slack2),
            'Slack_2_min'    :   np.min(slack2),
            'Slack_2_max'    :   np.max(slack2),
            'Slack_2_25p'    :   np.percentile(slack2, 25),
            'Slack_2_75p'    :   np.percentile(slack2, 75),
            'Slack_2_norm'    :   np.linalg.norm(slack2),
            "Slack_1_nonzero":   np.sum(abs(slack1)>=tol)
        }

    def stdObjM(self):
	    """
	    Standard deviation of normalized coefficients: f_i/Number_of_constraints
	    """
	    stdObjM= np.std(self.f/self.Aeq.shape[0],ddof=1)
	    return {'stdObjM': stdObjM}

    def stdObjN(self):
    	"""
    	#Standard deviation of ci/ni where ni denotes
	    #the number of nonzero entries in column i of A
	    """
    	n = self.Aeq.shape[1]
    	m = self.Aeq.shape[0]
    	fNew1 = np.zeros((1,n))
    	fNew2 = np.zeros((1,n))
    	for i in range(n):
            c = 0
            for j in range(m):
                if self.Aeq[j][i] != 0:
                    c += 1
            fNew1[0][i] = self.f[i]/c
            fNew2[0][i] = self.f[i]/np.sqrt(c)
    	stdObjN=np.std(fNew1,ddof=1)
    	stdObjRootN = np.std(fNew2,ddof=1)
    	return {'stdObjN':stdObjN, 'stdObjRootN':stdObjRootN}
    def AeqNormStats(self):
        """
        Distribution of normalized constraint matrix entries,
        Aij/bi: mean and std (only of elements where bi != 0)
        """

        n = self.Aeq.shape[1]
        m = self.Aeq.shape[0]
        AeqNorm=np.zeros((m,n))
        for i in range(m):
    	    for j in range(n):
                if self.beq[i]!=0:
                    AeqNorm[i][j]=self.Aeq[i][j]/self.beq[i]
        AeqNormMean=np.mean(AeqNorm)
        AeqNormStd=np.std(AeqNorm,ddof=1)
        return {'AeqNormMean':AeqNormMean,'AeqNormStd':AeqNormStd}
    def CVStats(self):
        """
        Variation coefficient of normalized absolute nonzero
        entries per row: mean and Std
        """
        AeqAbsolute=np.absolute(self.Aeq)
        var=variation(AeqAbsolute,axis=1)
        CVMean=np.mean(var)
        CVStd=np.std(var,ddof=1)
        return {'CVMean':CVMean,'CVStd':CVStd}
    def AOverb(self):
        """
        Min/max for ratios of constraint coeffs. to RHS: Min and Max
        ratios across positive and negative right-hand-sides
        """
        n = self.Aeq.shape[1]
        m = self.Aeq.shape[0]
        b=np.zeros((1,n))
        MinPos = math.inf
        MaxPos = -math.inf
        MinNeg = math.inf
        MaxNeg = -math.inf
        for i in range(m):
            if self.beq[i]>0:
                for j in range(n):
                    b[0][j]=self.Aeq[i][j]/self.beq[i]
                    MinTemp = np.amin(b)
                    MaxTemp = np.amax(b)
                if MinTemp<MinPos:
                    MinPos = MinTemp
                if MaxTemp>MaxPos:
                    MaxPos = MaxTemp
            elif self.beq[i]<0:
                for j in range(n):
                    b[0][j]=self.Aeq[i][j]/self.beq[i]
                    MinTemp1 = np.amin(b)
                    MaxTemp1 = np.amax(b)
                if MinTemp1<MinNeg:
                    MinNeg = MinTemp1
                if MaxTemp1>MaxNeg:
                    MaxNeg = MaxTemp1
        return {'MinPos':MinPos,'MaxPos':MaxPos,'MinNeg':MinNeg,'MaxNeg':MaxNeg}
    def OnetoAllA(self):
        """
        Min/max for one-to-all coeff ratios: The statistics are over the
        ratios of a variables coefficient, to the sum over all other variables
        coefficients, for a given constraint. Four versions of these ratios are
        considered: positive (negative) coefficient to sum of positive (negative)
        coefficients
        """
        Aeq = self.Aeq
        n = self.Aeq.shape[1]
        m = self.Aeq.shape[0]
        List1 = [] #pospos
        List2 = [] #posneg
        List3 = [] #negneg
        List4 = [] #negpos
        for i in range(m):
            a=Aeq[i][:]
            pos=a[a>0]
            neg=a[a<0]
            sumPos=np.sum(pos)
            sumNeg=np.sum(neg)
            sizeP = np.size(pos)
            sizeN = np.size(neg)
            c=np.zeros((1,sizeP))
            d=np.zeros((1,sizeP))
            e=np.zeros((1,sizeN))
            f=np.zeros((1,sizeN))
            #PosPos
            if sumPos == 0: #No positive numbers
                List1.append(0)
                List2.append(0)
            else:
                for j in range(sizeP):
                    if (sumPos-pos[j])==0: #only 1 positive
                        c[0][j]=pos[j]
                    else:
                        c[0][j]=pos[j]/(sumPos-pos[j]) #positive/positive
                    List1.append(c[0][j])
            #PosNeg
            for j in range(sizeP):
                if sumNeg == 0:
                    d[0][j]=-pos[j]
                elif sumPos != 0:
                    d[0][j]=pos[j]/sumNeg #positive/negative
                List2.append(d[0][j])
            #NegNeg
            if sumNeg == 0: #No negative numbers
                List3.append(0)
                List4.append(0)
            else:
                for j in range(sizeN):
                    if (sumNeg-neg[j])==0: #only 1 negative
                        e[0][j]=-neg[j]
                    else:
                        e[0][j]=neg[j]/(sumNeg-neg[j]) #negative/negative
                    List3.append(e[0][j])
            #NegPos
            for j in range(sizeN):
                if sumPos == 0:
                    f[0][j]=-pos[j]
                elif sumNeg != 0:
                    f[0][j]=neg[j]/sumPos #negative/positive
                List4.append(f[0][j])        
        MinPosPos=min(List1)
        MaxPosPos=max(List1)
        MeanPosPos=np.mean(List1)
        StdPosPos=np.std(List1,ddof=1)
        MinPosNeg=min(List2)
        MaxPosNeg=max(List2)
        MeanPosNeg=np.mean(List2)
        StdPosNeg=np.std(List2,ddof=1)
        MinNegNeg=min(List3)
        MaxNegNeg=max(List3)
        MeanNegNeg=np.mean(List3)
        StdNegNeg=np.std(List3,ddof=1)
        MinNegPos=min(List4)
        MaxNegPos=max(List4)
        MeanNegPos=np.mean(List4)
        StdNegPos=np.std(List4,ddof=1)
        
        return{'MinPosPos':MinPosPos,'MaxPosPos':MaxPosPos,
               'MinPosNeg':MinPosNeg,'MaxPosNeg':MaxPosNeg,
               'MinNegPos':MinNegPos,'MaxNegPos':MaxNegPos,
               'MinNegNeg':MinNegNeg,'MaxNegNeg':MaxNegNeg,
               'MeanPosPos':MeanPosPos, 'StdPosPos':StdPosPos,
               'MeanPosNeg':MeanPosNeg, 'StdPosNeg':StdPosNeg,
               'MeanNegNeg':MeanNegNeg, 'StdNegNeg':StdNegNeg,
               'MeanNegPos':MeanNegPos, 'StdNegPos':StdNegPos}    
    def VGraph(self, tol=1e-9):
        """
        Returns statistics of the variable graph. It is a simple graph where each vertex
        represent the variables. An edge between them implies that the variables
        appear in the same constraint ever.
        """
        v = self.f.size
        # Making the adjacency of variable graph
        VG = np.zeros((v, v))
        for i in range(v):
            for j in range(i+1):
                for k in self.Aeq:
                    if abs(k[i]) >= tol and abs(k[j]) >= tol:
                        VG[i, j] = 1
                        VG[j, i] = 1
                        break
        Variable_Node_vec = np.sum(VG, axis = 0)   # Node degree of each variable node
        return {
            'VG_V_mean':   np.mean(Variable_Node_vec),              # Mean degree of variable node
            'VG_V_std' :   np.std(Variable_Node_vec),               # Std. dev of degree of variable node
            'VG_V_min' :   np.min(Variable_Node_vec),               # Min degree of a variable node
            'VG_V_max' :   np.max(Variable_Node_vec),               # Max degree of a variable node
            'VG_V_25p' :   np.percentile(Variable_Node_vec, 25),    # 25th percentile degree of a variable node
            'VG_V_75p' :   np.percentile(Variable_Node_vec, 75),    # 75th percentile degree of a variable node
            'EdgeDens' :   np.sum(VG)*1.0/np.size(VG)               # Percentage of edges pressent
        }
    def getProbingFeatures(self, TL = 10):
        """
        Up to the time limit specified, gets the number of cuts of various types added by CPLEX.
        Also gives presolve information
        """
        if not self.Probed:
            self.MakeCplex()
            #dictionary to be returned
            D = {}
            c = self.CplexObjectMIP # Short name for the object
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
            # Computes relative gap (WHAT TO DO IF NO SOLUTION FOUND? Gets exception (try on enlight13))
            #relGap = c.solution.MIP.get_mip_relative_gap()
            self.Probingdict = D
            self.Probed = True
        else:
            D = self.Probingdict
        return D
    # Geometric features
    def geometricFeatures(self):
        Aeq = self.Aeq
        # We are interested in taking the scalar product of every column of Aeq with every column of Aeq.
        # This is precisely obtained by (Aeq^T)(Aeq)
        G = {}
        (m, n) = Aeq.shape
        nrm = np.linalg.norm(Aeq,axis=0)
        G['nrm_mean_Aeq'] = np.mean(nrm)
        G['nrm_sd_Aeq'] = np.std(nrm)
        G['nrm_coeffvar_Aeq'] = G['nrm_sd_Aeq']/G['nrm_mean_Aeq']
        A_norm = Aeq/nrm
        A_dot = A_norm.T.dot(A_norm)
        tole = 1e-9
        mask = (np.tril(np.ones((n, n))) - np.identity(n)) > tole
        scalar_vec = np.extract(arr = A_dot, condition = mask) # vector of scalar products 
        G['scalar_mean_Aeq'] = np.mean(scalar_vec)
        G['scalar_sd_Aeq'] = np.std(scalar_vec)
        G['scalar_var_Aeq'] = np.var(scalar_vec)
        G['scalar_coeffvar_Aeq'] = G['scalar_sd_Aeq']/G['scalar_mean_Aeq']
        angle_vec = np.arccos(scalar_vec)
        G['angle_mean_Aeq'] = np.mean(angle_vec)
        G['angle_sd_Aeq'] = np.std(angle_vec)
        G['angle_var_Aeq'] = np.var(angle_vec)
        G['angle_coeffvar_Aeq'] = G['angle_sd_Aeq']/G['angle_mean_Aeq']
        # Getting the same for Nonbasic tableaux
        self.LPSolve()
        Aeq = self.LPInfo["Tableaux_NB"].todense()
        (m, n) = Aeq.shape
        nrm = np.linalg.norm(Aeq,axis=0)
        G['nrm_mean_NB'] = np.mean(nrm)
        G['nrm_sd_NB'] = np.std(nrm)
        G['nrm_coeffvar_NB'] = G['nrm_sd_NB']/G['nrm_mean_NB']
        A_norm = Aeq/nrm
        A_dot = A_norm.T.dot(A_norm)
        mask = (np.tril(np.ones((n, n))) - np.identity(n)) > tole
        scalar_vec = np.extract(arr = A_dot, condition = mask) # vector of scalar products 
        G['scalar_mean_NB'] = np.mean(scalar_vec)
        G['scalar_sd_NB'] = np.std(scalar_vec)
        G['scalar_var_NB'] = np.var(scalar_vec)
        G['scalar_coeffvar_NB'] = G['scalar_sd_NB']/G['scalar_mean_NB']
        angle_vec = np.arccos(scalar_vec)
        G['angle_mean_NB'] = np.mean(angle_vec)
        G['angle_sd_NB'] = np.std(angle_vec)
        G['angle_var_NB'] = np.var(angle_vec)
        G['angle_coeffvar_NB'] = G['angle_sd_NB']/G['angle_mean_NB']
        return G
    # End of feature extraction
    #######################






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
    if Prob.name != '':
        M.set_problem_name(Prob.name)
    return M

def Py2CplexMIP(Prob):
    """
    Given a problem (MIP object)
        min c^T x subject to
        Ax = b
        x >= 0
        x_i \in \Z if i \in intcon
    returns a CPLEX model object 
    """
    M = Py2Cplex(Prob)
    types =  ['C' if i else 'I' for i in Prob.cont]
    M.variables.set_types(zip(range(Prob.f.size), types))
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
    # Continuous or integer?
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
                "name":M.get_problem_name(),
                "beq":beq,
                "cont":np.array(cont),
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
