from ReprVars import *
import sys


args = sys.argv

F_naam = args[1].split("_")[0]
if len(args)<2:
    print("Usage is python genFeat <MIPlist> <SolFolder> where <MIPlist> is name of a file containing a list of strings. For each string, ")
    print("MIP object would be made with <string>_Aeq.csv, <string>_beq.csv and so on")

path = "/home-4/ssankar5@jhu.edu/scratch/201804/"+F_naam+"/"
Sol_path = "/home-4/ssankar5@jhu.edu/scratch/201804/"+F_naam+"/"


FileObj = open(args[1], "r")
names = [line.rstrip() for line in FileObj.readlines()]

N_feat = 80 # Number of features

FeaturesAll = np.zeros((0, N_feat+3))

count = 0
for name in names:
    # Mixed integer program
    # Solution analysis
    Sol_value = np.genfromtxt(Sol_path+name+"_Sol.csv", delimiter = ",").squeeze()
    v_MIP = [] # List for the mixed integer program    
    LP = Sol_value[0]
    v_MIP.append(LP) # LP relaxed solution
    GMI = Sol_value[2]
    v_MIP.append((GMI-LP)/abs(LP)) # Improvement after adding all GMI
    BestObj = max(Sol_value[5], Sol_value[6],Sol_value[9],Sol_value[10], #2row X, 2row XG, 2row GX, 2row GXG
                Sol_value[13], Sol_value[14], Sol_value[17], Sol_value[18], # 5row X, 5row XG, 5row GX, 5row GXG
                Sol_value[21], Sol_value[22], Sol_value[25], Sol_value[26]) # 10row X, 10row XG, 10row GX, 10row GXG
    v_MIP.append((BestObj - GMI)/(GMI - LP)) # Beta as defined in Basu and Sankaranarayanan (2018)
    v_MIP = np.array(v_MIP)
    # Problem data analysis
    data = {
            'name':name+"_MILP",
            'f':path+name+"_obj.csv",
            "Aeq":path+name+"_Aeq.csv",
            "beq":path+name+"_beq.csv",
            "cont":path+name+"_cont.csv"
            }
    M = MIP(form = 1,
            data = data,
            filenames = True,
            delimiter = ","
            )
    v = M.features(returnAsVect = True)
    v = np.concatenate((v, v_MIP)).reshape(1, N_feat+3) # N_feat features from MIP class, 3 more from solution file
    FeaturesAll = np.concatenate((FeaturesAll, v), axis = 0)
    # Pure integer program
    v_PIP = [] # List for the pure integer program    
    LP = Sol_value[0]
    v_PIP.append(LP) # LP relaxed solution
    GMI = Sol_value[1]
    v_PIP.append((GMI-LP)/abs(LP)) # Improvement after adding all GMI
    BestObj = max(Sol_value[3], Sol_value[4],Sol_value[7],Sol_value[8],
                Sol_value[11], Sol_value[12], Sol_value[15], Sol_value[16],
                Sol_value[19], Sol_value[20], Sol_value[23], Sol_value[24])
    v_PIP.append((BestObj - GMI)/(GMI - LP)) # Beta as defined in Basu and Sankaranarayanan (2018)
    v_PIP = np.array(v_PIP)
    # Problem data analysis
    M_pure = MIP(form = 1,
            data = data,
            filenames = True,
            delimiter = ","
            )
    M_pure.name = name+"_PILP"
    M_pure.cont = M_pure.cont*0         # None of them is a continuoous variable
    v = M_pure.features(returnAsVect = True)
    v = np.concatenate((v, v_PIP)).reshape(1, N_feat+3) # N_feat features from MIP class, 3 more from solution file
    FeaturesAll = np.concatenate((FeaturesAll, v), axis = 0)
    # Display completion
    print("Done for problem number: "+name)
    count = count+1
    if count % 5 == 0:
        np.savetxt("Feat_100/F_"+args[1]+".csv", FeaturesAll, fmt="%s")

np.savetxt("Feat_100/F_"+args[1]+".csv", FeaturesAll, fmt="%s")
