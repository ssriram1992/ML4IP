import numpy as np
import scipy as sp


class MIP:
    """
    A class to represent Mixed-integer-programs in standard form
    Standard form is the follows
    \max f^Tx subject to 
    Ax = b
    x >= 0
    x \in \Z for x\ not\in cont
    """
    def __init__(self):
        """
        Empty initialization
        """
        self.f = numpy.array([[]]).reshape((0,1))
        self.A = numpy.array([[]]).reshape((0,0))
        self.b = numpy.array([[]]).reshape((0,1))
        self.cont = numpy.array([])
    def __init__(self, f, A, b, cont, filenames = 0, delimiter = ','):
        """
        Can enter f, A, b, cont either as matrices of appropriate size (filenames = 0)
        Or as names to csv files, from which the data can be fetched. (filenames = 1)
        """
        if filenames:
            self.f = np.genfromtxt(f, delimiter=delimiter)
            self.A = np.genfromtxt(A, delimiter=delimiter)
            self.b = np.genfromtxt(b, delimiter=delimiter)
            self.cont = np.genfromtxt(cont, delimiter=delimiter)
        else:
            self.f = f
            self.A = A
            self.b = b
            self.cont = cont
    def size(self):
        """
        Returns the number of variables, number of constraints and number of integer 
        variables in the IP
        """
            nVar = np.size(self.f)
            nCons = np.size(self.b)
            nInt = nVar - np.sum(self.cont)
            return {'nVar':nVar, 'nCons':nCons, 'nInt':nInt}
