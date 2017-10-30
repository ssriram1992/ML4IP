import cplex

def Py2Cplex(MIP):
	"""
	Given a problem (MIP object)
		min c^T x subject to
		Ax = b
		x >= 0
		x_i \in \Z if i \in intcon
	returns a CPLEX model object 
	"""
	pass

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