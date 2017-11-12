import numpy as np
import scipy as sp



# No class, writing functions 

def addCuts(inMIP N_in, cutA, cutb):
    """
    outMIP =  addCuts(inMIP, N_in, cutA, cutb)
    INPUT:
    inMIP: An object of MIP class. Hence constraints inMIP.f, inMIP.A, inMIP.b in standard form. 
    N_in: Indices corresponding to non-basic variables 
    cutA: LHS matrix for the cut inequalities
    cutb: RHS vector for the cut inequalities
    cutA x >= cutb should be valid for the MIP
    Output:
    outMIP: An object of MIP class with the original constraints as well as the cuts. 
    Slacks etc are added and the problem will be in standard form. 
    """
    f = inMIP.f
    A = inMIP.Aeq
    b = inMIP.beq
    nCuts = cutA.shape[0]
    nVars = f.shape[0]
    nCons = b.shape[0]
    cont = inMIP.cont
    # The new objective function. objective 0s for the slacks
    fnew = np.zeros((nVars+nCuts,1)) 
    fnew[np.arange(nVars),0] = f
    # New Aeq matrix
    # Number of constraints and variables both increase by nCuts
    Anew = np.zeros((nCons+nCuts, nVars+nCuts))
    Anew[0:nCons, 0:nVars] = A;
    Anew[nCons:, N_in] = -cutA #cut should go to the N_in columns
    Anew[nCons:,nVars:] = np.identity(nCuts) #Identity corresponding to slacks
    # b
    bnew = np.zeros((nCons+nCuts,1))
    bnew[0:nCons,0] = b;
    bnew[nCons:,0] = -cutb
    # continuous or integer?
    contnew = np.zeros(fnew.shape)
    contnew[0:nVars,0] = cont # same integrality constraints as in older problem, all slacks assumed to be continuous
    # Creating the class
    outMIP = MIP( form = 1, # Standard form
        data = {'f' : f, 'Aeq' : Aeq, 'beq' : beq, 'cont' : cont}, # MIP Data
        filenames = False # Defining from actual variables
        )
    return outMIP







def GMI(N, b, rows, cont_NB):
    """
    (A_GMI, b_GMI) = GMI(N, b, rows, cont_NB)
    INPUT ARGUMENTS:
    N is the non-basic matrix from the optimal simplex tableaux
    b is the LP relaxed solution obtained
    rows is a 0/1 vector saying which rows can be used to create GMI cuts 
        (note this has to be a subset of integer variables that turned out continuous in LP relaxation.
        Only then the cut is feasible)
    cont_NB says which of the non-basics are continuous to generate appropriate cut-coefficients
    OUTPUT ARGUMENTS:
    A_GMI, b_GMI is a matrix and vector respectively. They are such that, 
    A_GMI x >= b_GMI is valid for the MIP
        (note the inequality is >= and not <=. This is consistent with the standard form the cuts are written in)
    """
    b_GMI = np.ones((np.sum(rows),1)) # RHS is just a vector of ones
    f0 = b - np.floor(b) # This ensures that b is [0,1]^n hypercube.
    nVar = N.shape[1] # Number of columns in non-basic matrix  =  number of variables
    int_NB = set(np.arange(nVar))-set(list(np.where(cont_NB)[0])) # Not continuous => integer
    find_rows = np.where(rows)[0] # Find the indices corresponding to rows whose GMI is to computed 
    A_GMI = np.zeros(N.shape)
    fj = N - np.floor(N) # Getting the fractional part of non-basic matrix
    # For each row of the non-basic matrix
    for i in np.arange(N.shape[0]):
        # Check if a cut has to be computed at all by the input argument "rows"
        if i not in find_rows:
            continue
        # Following equation 5.31 in Conforti, Cornuejols, Zambelli for each column
        for j in np.arange(nVar):            
            if j in int_NB:
                # Following for integer variables
                if fj[i,j] < f0[i]:
                    A_GMI[i,j] = fj[i,j]/f0[i];
                else:
                    A_GMI[i,j] = (1-fj[i,j])/[1-f0[i]]
            else:
                # Following for continuous variables
                if N[i,j]>0:
                    A_GMI[i,j] = N[i,j]/f0[i]
                else: 
                    A_GMI[i,j] = -N[i,j]/(1-f0[i])
    # Eliminating the rows for which cuts were not computed
    A_GMI = A_GMI[find_rows,:]
    return (A_GMI, b_GMI)

def intRows(x_B, int_B, epsilon = 1e-12):
    """
    INPUT
    x_B:     Current basic solution
    int_B:   0/1 vector, same shape as x_B, indicating which of them should be 
    epsilon: tolerance for considering a number as integer
    OUTPUT
    badrows: returns 0/1 vector indicating the set of basics 
            that should have been integers but not integers. 
    """
    t1 = np.remainder(x_B, 1)>=epsilon  # Is it epsilon farther from an integer?
    t2 = int_B # is it expected to be an integer?
    return np.logical_and(t1, t2) # Satisfies both the above conditions?

def Rows4Xcut(x_B, nRows, nCuts, intVar, n_badrow):
    """
    INPUTS:
    x_B = current basic solution. This also tells how many rows are available to pick from
    nRows = Number of rows to pick
    nCuts = Number of cuts to be added.
    intVar = 1/0 vector with same size as x_B saying which of the rows should be
    integers. Picking is done only from these rows.
    n_badrow = Minimum number of badrows to be selected for each cut. 
            (nRows - n_badrow) will be chosen from goodrows
            (badrow = should be integer, but LP relaxation gave non-integer)
            (goodrow = should be integer, but LP relaxation gave integer row)
    OUTPUTS:
    All outputs are nRows x nCuts sparse matrices
    ans["RowMat"] : 0/1 matrix indicating which row to pick for which cut
    ans["muMat"] : elements in [0,1) giving mu values for each cut. mu sums to 1 along rows
    ans["fMat"] : elements in [0,1) giving f values for each cut.
    """
    nCons = x_B.shape[0] # Number of constraint rows to pick from 
    badrow = intRows(x_B, intVar)
    goodrow = np.logical_and(
        np.logical_not(badrow), 
        intVar
        )
    # Note that badrow union goodrow = intVar
    if np.sum(intVar) < nRows: # Number of integer rows < number of rows to pick?
        print("Too few rows to pick from") # Too bad! Can't do that!
        return 
    if np.sum(badrow) < n_badrow: # If LP solution has only 1 bad row, and you are trying to pick 2 bad rows?
        print ("Too few bad rows")  # Too bad! Can't do that!
        return
    if np.sum(goodrow) < (nRows-n_badrow): # If there are too few goodrows
        n_badrow = nRows - np.sum(goodrow) # We have to pick more badrows in the cuts. 
    # Finding indices of the 0/1 vectors
    good = np.where(goodrow)[0]
    bad = np.where(badrow)[0]
    ints = np.where(intVar)[0]
    # Since we will construct sparse matrices, generating row and column indices
    # Note, we are dealing with objects of class scipy.sparse.csc_matrix 
    # Documentation in https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix
    # This form of sparse matrix is fastest for column splicing which is what we need
    t_RowInd = np.zeros((nRows*nCuts,))
    t_ColInd = np.zeros((nRows*nCuts,))
    muVals = np.zeros((nRows*nCuts,))
    fVals = np.zeros((nRows*nCuts,))
    # Generating values for each cut
    for i in np.arange(nCuts):
        t2 = np.random.permutation(bad)[0:n_badrow] # Pick n_badrow numbers of badrow in random
        t3 = np.random.permutation(good)[0:(nRows-n_badrow)] # Pick remaining goodrows
        t1 = np.concatenate((t2,t3))
        t1.sort() # Just to be clean
        t_RowInd[ (i*nRows)+np.arange(nRows) ] = t1 # picks rows in t1
        t_ColInd[ (i*nRows)+np.arange(nRows) ] = i # for i-th cut
        # Selectitng mu
        muChoice = np.random.rand(nRows, )
        muChoice = muChoice/np.sum(muChoice) # normalizing mu to sum to 1
        muVals[ (i*nRows)+np.arange(nRows) ] = muChoice
        # selecting f
        fChoice = np.random.rand(nRows, )
        fVals[ (i*nRows)+np.arange(nRows) ] = fChoice
    # Actually creating the sparse matrices now
    RowMat = sp.sparse.csc_matrix((np.ones(t_RowInd.shape),(t_RowInd, t_ColInd)),shape = (nCons, nCuts))
    muMat = sp.sparse.csc_matrix((muVals, (t_RowInd, t_ColInd)),shape = (nCons, nCuts))
    fMat = sp.sparse.csc_matrix((fVals, (t_RowInd, t_ColInd)),shape = (nCons, nCuts))
    ans = dict()
    ans["RowMat"] = RowMat
    ans["muMat"] = muMat
    ans["fMat"] = fMat
    return ans



        

def XGauge(mu, b, x):
    """
    gauge = XPolyGauge(mu, b, x)
    mu: vector in [0,1]^n whose components sum to 1. Parameterizes the cross polytope
    b: vector in [0,1]^n fractional part of the LP solution
    x: non-basic column. Gauge will be evaluated at this point

    Returns the gauge function given \mu and b. b should be non-negative 
    and each coordinate should be less than 1.
    mu should sum to 1.
    The center of the polytope is assumed to coincide with origin.
    """    
    # dimension we are in
    n = mu.shape[0]
    # reshaping to nx1 vectors as opposed to pythons (n,) shaped tuple, just in case
    mu = mu.reshape((n,1))
    b = b.reshape((n,1))
    x = x.reshape((n,1))
    a = np.zeros((n,1))
    for i in np.arange(n):
        if x[i] >= 0:
            a[i,0] = mu[i,0]/b[i,0]
        else:
            a[i,0] = mu[i,0]/(b[i,0]-1)
    gauge = a.T.dot(x)[0,0] # [0,0] to return as a scalar, rather than 1x1 matrix
    XG = dict()
    XG["gauge"] = gauge
    XG["normal"] = a
    return XG


def XLift(N, b, RowMat, muMat, cont_NB):
    """
    (A_Xcut, b_Xcut) = GXLift(N, b, RowMat, muMat, fMat, cont_NB)
    INPUTS:
    N       Nonbasic matrix
    b       LP solution
    muMat   A Nrow x Ncut matrix with each column being a mu vector for a cut. Sum across rows should be 1
    cont_NB boolean vector that says which of the non-basics are continuous to generate appropriate cut-coefficients
    """
    nCuts = RowMat.shape[1] 
    nVars = N.shape[1]
    # Ensure b is  matrix rather than a python tuple
    b = b.reshape((b.shape[0],1))
    # Initializing the cuts
    A_Xcut = np.zeros((nCuts, nVars))
    b_Xcut = np.ones((nCuts,1))
    cont = set(list(np.where(cont_NB)[0]))
    # One iteration of this loop for each cut
    for cut in np.arange(nCuts):
        # Get the rows of the non-basic used in this cut
        rows = RowMat[:, cut].astype(bool)
        # Bring "b" into the unit hypercube [0,1]^n
        bb = b[rows,0] - np.floor(b[rows,0])
        mu = muMat[rows, cut]
        # reshaping them into nx1 vectors
        mu = mu.reshape((mu.shape[0],1))
        bb = bb.reshape((bb.shape[0],1))
        # Using the function defined before to get the facets of the cross polytope
        ### a_Actual = GXGaugeA(mu, f, bb)
        # For each column of non-basic, creating lifting or gauge
        for var in np.arange(nVars):
            # Non-basic
            x = N[rows, var]
            x = x.reshape((x.shape[0],1))
            if var in cont:
                # If it is continuous variable column, then calculate gauge
                # else the lifting
                A_Xcut[cut, var] = XGauge(mu, b, x)["gauge"]
            else:
                # "else the lifting" part
                n = np.shape(mu)[0]     # Dimensionality of the crosspolytope
                xfrac = x - np.floor(x) # Bringing x into [0,1]^n hyper cube
                primshift = (xfrac > bb)*1 # Is x outside [b-1, b]^n hypercube?
                xmid = xfrac - primshift # Bring x into [b-1, b]^n hypercube
                lifting = 1 # Initialization
                # Calculating the best gauge in each drection
                for direction in np.arange(n):
                    # Solving a 1d convex IP in each dimension
                    shift = np.zeros((n,1))
                    shift[direction,0] = 1 # shift is a coordinate unit vector now
                    lb = np.ceil((bb[direction, 0]-1)/mu[direction, 0])
                    ub = np.floor((bb[direction, 0])/mu[direction, 0])
                    while True:
                        mid = np.round((lb+ub)/2.0,0)
                        t0 = XGauge(mu, b, xmid+mid*shift)
                        t1 = t0["gauge"]
                        t2 = t0["normal"]
                        ### temp = a_Actual.dot(xmid+mid*shift) # scalar product with all normals
                        ### t2 = np.argmax(temp) # which'th normal achieves the gauge
                        ### t1 = temp[t2]        # the gauge
                        if t2[direction, 0] < 0:
                            lb = mid
                        else:
                            ub = mid
                        if ub-lb <= 1:
                            break
                    if lb == ub:
                        lifting = np.minimum(lifting, t1)
                    else:
                        lifting = np.min([lifting, 
                            XGauge(mu, b, xmid+lb*shift)["gauge"],
                            XGauge(mu, b, xmid+ub*shift)["gauge"]])
                    # End of direction for loop
                A_Xcut[cut, var] = lifting
                # End of else for lifting of integer variable
            # End of var for loop
        print('cut ' + str(cut) + 'generated' )
        # End of cut for loop
    return (A_Xcut, b_Xcut)
    # End of lifting function


def GXGaugeA(mu, f, b):
    """
    A = GXGaugeA(mu, f, b)
    mu:  vector in [0,1]^n whose components sum to 1. Parameterizes the cross polytope
    f: Location of the crosspolytope's center, relative to the origin
    b: vector in [0,1]^n fractional part of the LP solution

    Given a generalized cross-polytope's description in terms of mu, f and b, returns
    matrix A, containing the normals to the facets of the polytope.
    """
    n = mu.shape[0]
    mu = mu.reshape((n,1))
    f = f.reshape((n,1))
    b = b.reshape((n,1))
    t1 = np.arange(2**n).reshape((2**n,1)) # 0 to (2^n)-1 ix contains normals to ine./
    t2 = np.power(2.0, np.arange(-(n-1),1)).reshape((1,n)) #1/2^n vectorrs
    AllVecs = np.remainder(np.floor(t1.dot(t2)),2) # Generate all binary vectors of size n
    # Matrix contains normals to each face of Xpolytope had the center been origin
    aMat = np.divide(np.squeeze(mu), b.T-f.T-AllVecs)
    one_plus_a_dot_f = 1 + aMat.dot(f)    
    if np.min(one_plus_a_dot_f) <= 0:
        print(one_plus_a_dot_f)
        print('Potential problems')
    a_Actual = np.divide(aMat, one_plus_a_dot_f)
    return a_Actual


def GXLift(N, b, RowMat, muMat, fMat, cont_NB):
    """
    (A_GXcut, b_GXcut) = GXLift(N, b, RowMat, muMat, fMat, cont_NB)
    INPUTS:
    N       Nonbasic matrix
    b       LP solution
    muMat   A Nrow x Ncut matrix with each column being a mu vector for a cut. Sum across rows should be 1
    fMat    A Nrow x Ncut matrix with each columb being the location of center for a cut
    cont_NB says which of the non-basics are continuous to generate appropriate cut-coefficients
    """
    nCuts = RowMat.shape[1] 
    nVars = N.shape[1]
    # Ensure b is  matrix rather than a python tuple
    b = b.reshape((b.shape[0],1))
    # Initializing the cuts
    A_GXcut = np.zeros((nCuts, nVars))
    b_GXcut = np.ones((nCuts,1))
    cont = set(list(np.where(cont_NB)[0]))
    # One iteration of this loop for each cut
    for cut in np.arange(nCuts):
        # Get the rows of the non-basic used in this cut
        rows = RowMat[:, cut].astype(bool)
        # Bring "b" into the unit hypercube [0,1]^n
        bb = b[rows,0] - np.floor(b[rows,0])
        # Ensure that the origin and the center of the GX-polytope are in the same hypercube
        f = fMat[rows, cut] - np.ceil(fMat[rows, cut]-bb)
        mu = muMat[rows, cut]
        # reshaping them into nx1 vectors
        mu = mu.reshape((mu.shape[0],1))
        f = f.reshape((f.shape[0],1))
        bb = bb.reshape((bb.shape[0],1))
        # Using the function defined before to get the facets of the cross polytope
        a_Actual = GXGaugeA(mu, f, bb)
        # For each column of non-basic, creating lifting or gauge
        for var in np.arange(nVars):
            # Non-basic
            x = N[rows, var]
            x = x.reshape((x.shape[0],1))
            if var in cont:
                # If it is continuous variable column, then calculate gauge
                # else the lifting
                A_GXcut[cut, var] = np.max(a_Actual.dot(x))
            else:
                # "else the lifting" part
                n = np.shape(mu)[0]     # Dimensionality of the crosspolytope
                xfrac = x - np.floor(x) # Bringing x into [0,1]^n hyper cube
                primshift = (xfrac > bb)*1 # Is x outside [b-1, b]^n hypercube?
                xmid = xfrac - primshift # Bring x into [b-1, b]^n hypercube
                lifting = 1 # Initialization
                # Calculating the best gauge in each drection
                for direction in np.arange(n):
                    # Solving a 1d convex IP in each dimension
                    shift = np.zeros((n,1))
                    shift[direction,0] = 1 # shift is a coordinate unit vector now
                    lb = np.ceil((bb[direction, 0]-1)/mu[direction, 0])
                    ub = np.floor((bb[direction, 0])/mu[direction, 0])
                    while True:
                        mid = np.round((lb+ub)/2.0,0)
                        temp = a_Actual.dot(xmid+mid*shift) # scalar product with all normals
                        t2 = np.argmax(temp) # which'th normal achieves the gauge
                        t1 = temp[t2]        # the gauge
                        if a_Actual[t2, direction] < 0:
                            lb = mid
                        else:
                            ub = mid
                        if ub-lb <= 1:
                            break
                    if lb == ub:
                        lifting = np.minimum(lifting, t1)
                    else:
                        lifting = np.min([lifting, 
                            np.max(a_Actual.dot(xmid+lb*shift)),
                            np.max(a_Actual.dot(xmid+ub*shift)) ])
                    # End of direction for loop
                A_GXcut[cut, var] = lifting
                # End of else for lifting of integer variable
            # End of var for loop
        print('cut ' + str(cut) + 'generated' )
        # End of cut for loop
    return (A_GXcut, b_GXcut)
    # End of lifting function