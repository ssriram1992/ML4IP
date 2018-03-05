import numpy as np
import scipy as sp
from CPLEXInterface import *

# Making changes in Sri branch
class MIP:
    """
    A class to represent Mixed-integer-programs in standard form
    Standard form is the follows
    \min f^Tx subject to
    Aeq x = beq
    x >= 0
    x \in \Z for x\ not\in cont

    Flags:
    # These are binary falgs which are switched when a computation heavy operation is done
    # so that the operation need not be repeated unnecessarily.
    # Do not alter them if unsure.
    CplexMade
        True if corresponding CPLEX Object is made
    LPSolved
    GraphDrawn
    TableGraphDrawn

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

    Methods of use:
    MIP.write(name, path="./")
        Writes 4 csv files in the given path with the name prefix given, saving Aeq, beq, obj and cont
    MIP.MakeCplex()
        Makes a Cplex object of LP relaxation of the given MIP instance and saves it.
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
        Returns LP objective value
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
                                            tableaux = True,
                                            basic = True,
                                            TablNB = True,
                                            precission = 13
                                        )
    # End of general methods
    #######################
    # Feature extraction
    def size(self):
        """
        Returns the number of variables, number of constraints and number of integer
        variables in the IP
        """
        nVar = np.size(self.f)
        nCons = np.size(self.beq)
        nInt = nVar - np.sum(self.cont)
        return {'nVar':nVar, 'nCons':nCons, 'nInt':nInt}
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
        return LPInfo["Objective"]
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
        LPSol = LPInfo["Solution"].squeeze()
        slack2 = (LPSol - np.floor(LPSol))*(1-self.cont)
        slack1 = slack2[cont==0]
        return {
            'Slack_1_mean'   :   np.mean(slack1),
            'Slack_1_std'    :   np.std(slack1),
            'Slack_1_min'    :   np.min(slack1),
            'Slack_1_max'    :   np.max(slack1),
            'Slack_1_25p'    :   np.percentile(slack1, 25),
            'Slack_1_75p'    :   np.percentile(slack1, 75),
            'Slack_1_norm'    :   np.linalg.norm(slack1),
            'Slack_2_mean'   :   np.mean(slack2),
            'Slack_2_std'    :   np.std(slack2),
            'Slack_2_min'    :   np.min(slack2),
            'Slack_2_max'    :   np.max(slack2),
            'Slack_2_25p'    :   np.percentile(slack2, 25),
            'Slack_2_75p'    :   np.percentile(slack2, 75),
            'Slack_2_norm'    :   np.linalg.norm(slack2),
            "Slack_1_nonzero":   np.sum(abs(slack1)>=tol)
        }
    def VGraph(self, tol=1e-9):
        """
        Returns statistics of the variable graph.
        """
        v = self.f.size
        # Making the adjacency of variable graph
        VG = np.zeros((v, v))
        for i in range(v):
            for j in range(i+1):
                for k in Aeq:
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



    # End of feature extraction
    #######################





class VarConstGraph:
    def __init__(self):
        self.Adj = np.array([[]]).reshape((0,0))
    def __init__(self, Adj):
        self.Adj = Adj
