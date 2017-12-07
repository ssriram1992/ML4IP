import numpy as np
import scipy as sp


# Making changes in Sri branch
class MIP:
    """
    A class to represent Mixed-integer-programs in standard form
    Standard form is the follows
    \min f^Tx subject to
    Aeq x = beq
    x >= 0
    x \in \Z for x\ not\in cont
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
        self.LPSolved = False                     # Boolean to store if LP relaxation is solved
        self.GraphDrawn = False                   # Boolean to store if the Variable Constraint Graph is drawn
        self.TableGraphDrawn = False              # Boolean to store if the Variable Constraint Graph is drawn for simplex tableaux
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
    def write(self, name):
        np.savetxt(name + '_Aeq.csv', self.Aeq, delimiter = ',')
        np.savetxt(name + '_beq.csv', self.beq, delimiter = ',')
        np.savetxt(name + '_obj.csv', self.f, delimiter = ',')
        np.savetxt(name + '_cont.csv', self.cont, delimiter = ',')
    def size(self):
        """
        Returns the number of variables, number of constraints and number of integer
        variables in the IP
        """
        nVar = np.size(self.f)
        nCons = np.size(self.beq)
        nInt = nVar - np.sum(self.cont)
        return {'nVar':nVar, 'nCons':nCons, 'nInt':nInt}
    def createTableaux():
        """
        Function, that uses the
        information in f, Aeq, beq and solves the LP relaxation of the problem.
        The function should return LP solution vector, LP objective value,
        and the set of optimal Basic Variables/Non-basic variables (so that we
        can calculate the tableaux externally). Optionally, the tableaux can be
        directly returned from this function.
        """
        pass # TODO
    def drawVarConstGraph(self, forTableaux = 0):
        """
        Draws the variable constraint Graph either for the original problem.
        Or for the simplex tableaux
        """
        if forTableaux:
            pass #TODO
        else:
            VCG = VarConstGraph(self.Aeq != 0)


class VarConstGraph:
    def __init__(self):
        self.Adj = np.array([[]]).reshape((0,0))
    def __init__(self, Adj):
        self.Adj = Adj
