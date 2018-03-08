def stdObjM(self):
  """
  Standard deviation of normalized coefficients: f_i/Number_of_constraints
  """
  stdObjM= np.std(self.f/self.Aeq.shape[0],ddof=1)
  return {"stdObjM": stdObjM}
  

def stdObjN(self):
  """
  #Standard deviation of ci/ni where ni denotes 
  #the number of nonzero entries in column i of A
  """
  n = self.Aeq.shape[1]
  m = self.Aeq.shape[0]
  fNew1=np.zeros((n,1))
  fNew2=np.zeros((n,1))
  for i in range(n):
      c = 0
      for j in range(m):
          if self.Aeq[j][i] != 0:
              c += 1
      fNew1[i][0] = self.f[i][0]/c
      fNew2[i][0] = self.f[i][0]/np.sqrt(c)
  stdObjN=np.std(fNew1,ddof=1)
  stdObjRootN = np.std(fNew2,ddof=1)
  return {'stdObjN':stdObjN, 'stdObjRootN':stdObjRootN}




def AeqNormStats(self):
  """
  Distribution of normalized constraint matrix entries, 
  Aij/bi: mean and std (only of elements where bi != 0
  """
  n = self.Aeq.shape[1]
  m = self.Aeq.shape[0]
  AeqNorm=np.zeros((m,n))
  for i in range(m):
      for j in range(n):
          if self.beq[i][0]!=0:
              AeqNorm[i][j]=self.Aeq[i][j]/self.beq[i][0]
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
      if self.beq[i][0]>0:
          for j in range(n):
              b[0][j]=self.Aeq[i][j]/self.beq[i][0]
              MinTemp = np.amin(b)
              MaxTemp = np.amax(b)
          if MinTemp<MinPos:
              MinPos = MinTemp
          if MaxTemp>MaxPos:
              MaxPos = MaxTemp
      elif self.beq[i][0]<0:
          for j in range(n):
              b[0][j]=self.Aeq[i][j]/self.beq[i][0]
              MinTemp = np.amin(b)
              MaxTemp = np.amax(b)
          if MinTemp<MinNeg:
              MinNeg = MinTemp
          if MaxTemp>MaxNeg:
              MaxNeg = MaxTemp
  return {'MinPos':MinPos,'MaxPos':MaxPos,'MinNeg':MinNeg,'MaxNeg':MaxNeg}
  
  

def OnetoAllA(self):
  """  
  Min/max for one-to-all coeff ratios: The statistics are over the 
  ratios of a variable’s coefﬁcient, to the sum over all other variables’ 
  coefﬁcients, for a given constraint. Four versions of these ratios are 
  considered: positive (negative) coefﬁcient to sum of positive (negative) 
  coefﬁcients
  """
  n = self.Aeq.shape[1]
  m = self.Aeq.shape[0]
  
  MinPosPos = math.inf
  MaxPosPos = -math.inf
  MinPosNeg = math.inf
  MaxPosNeg = -math.inf
  MinNegPos = math.inf
  MaxNegPos = -math.inf
  MinNegNeg = math.inf
  MaxNegNeg = -math.inf

  for i in range(m):
      a=self.Aeq[i][:]
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
      for j in range(sizeP):
          if (sumPos-pos[j])==0:
              c[0][j]=pos[j]
          else:
              c[0][j]=pos[j]/(sumPos-pos[j]) #positive/positive
          d[0][j]=pos[j]/sumNeg #positive/negative
          MinPosTemp=np.amin(c)
          MaxPosTemp=np.amax(c)
          MinNegTemp=np.amin(d)
          MaxNegTemp=np.amax(d)
      if MinPosTemp<MinPosPos:
          MinPosPos = MinPosTemp
      if MaxPosTemp>MaxPosPos:
          MaxPosPos = MaxPosTemp
      if MinNegTemp<MinPosNeg:
          MinPosNeg = MinNegTemp
      if MaxNegTemp>MaxPosNeg:
          MaxPosNeg = MaxNegTemp
      for j in range(sizeN):
          if (sumNeg-neg[j])==0:
              e[0][j]=neg[j]
          else:
              e[0][j]=neg[j]/(sumNeg-neg[j]) #negative/negative
          f[0][j]=neg[j]/sumPos #negative/positive
          MinPosTemp=np.amin(f)
          MaxPosTemp=np.amax(f)
          MinNegTemp=np.amin(e)
          MaxNegTemp=np.amax(e)
      if MinPosTemp<MinNegPos:
          MinNegPos = MinNegTemp
      if MaxPosTemp>MaxNegPos:
          MaxNegPos = MaxPosTemp
      if MinNegTemp<MinNegNeg:
          MinNegNeg = MinNegTemp
      if MaxNegTemp>MaxNegNeg:
          MaxNegNeg = MaxNegTemp
  return{'MinPosPos':MinPosPos,'MaxPosPos':MaxPosPos,'MinPosNeg':MinPosNeg,
         'MaxPosNeg':MaxPosNeg,'MinNegPos':MinNegPos,
         'MaxNegPos':MaxNegPos,'MinNegNeg':MinNegNeg,
         'MaxNegNeg':MaxNegNeg)
