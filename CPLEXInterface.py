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
	# Note integer/continuos detail is not passed as CPLEX will solve this as
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