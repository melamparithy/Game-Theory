import numpy as np
import sys
# import copy

# n1 = 3
# n2 = 2
# u = np.array([1, 1, 0, 2, 0, 2, 1, 1, 0, 3, 2, 0]).reshape(-1, 2)

# n1 = 2
# n2 = 2
# u = np.array([3, 1, 1, 2, 1, 3, 3, 1]).reshape(-1, 2)

# n1 = 2
# n2 = 2
# u = np.array([2, 1, 0, 0, 0, 0, 1, 2]).reshape(-1, 2)

# n1 = 2
# n2 = 2
# u = np.array([2, 0, 0, 2, 0, 6, 6, 0]).reshape(-1, 2)

# n1 = 2
# n2 = 2
# u = np.array([2, 2, 1, 6, 6, 1, 5, 5]).reshape(-1, 2)

# n1 = 3
# n2 = 2
# u = np.array([3, 3, 2, 2, 0, 3, 3, 2, 5, 6, 6, 1]).reshape(-1,2)

n1 = int(input())
n2 = int(input())
u = np.array(list(map(float, input().split()))).reshape(-1,2)

flag = 1
minval = sys.float_info.max

for i in np.nditer(u):
    if i < 0:
        flag = 0
    minval = i if i < minval else minval

if not flag:
    u += -minval

# print(u)

A = u[:, 0].reshape(n2,n1).T
B = u[:, 1].reshape(n2,n1).T

# print(A, B, sep='\n')

P = np.hstack([B.T, np.eye(n2), np.ones(n2).reshape(-1,1)])
Q = np.hstack([np.eye(n1), A, np.ones(n1).reshape(-1,1)])

# print("####################")

# print(P, Q, sep="\n")


def integerPivot(col, tableau, nrows, basis):
    argmin= -1
    minval = sys.float_info.max

    for j in range(nrows):
        if (tableau[j, col] > 0 and tableau[j,-1] / tableau[j,col] < minval):
            minval = tableau[j,-1] / tableau[j,col]
            argmin = j
    # print(f'argmin is: {argmin}, {minval}')
    for j in range(nrows):
        if j != argmin:
            factor = tableau[j,col]
            tableau[j] = tableau[j]*tableau[argmin,col] - tableau[argmin]*factor
    
    label = basis[argmin]
    basis[argmin] = col
    return label

def lemkeHowson(n1, n2, startLabel, tableauP, tableauQ, chance):
    labelsRowPl = np.hstack([np.arange(n1), -np.ones(n2)])
    labelsColPl = np.hstack([-np.ones(n1), np.arange(n1,n1+n2)])

    basisRow = {}
    for i in range(n2):
        basisRow[i] = n1 + i 
    
    basisCol = {}
    for i in range(n1):
        basisCol[i] = i

    tableau = tableauQ if chance else tableauP
    nrows = n1 if chance else n2
    basis = basisCol if chance else basisRow

    duplabel = integerPivot(startLabel, tableau, nrows, basis)
    # print(duplabel)
    # print(tableauP)
    # print(basisRow)
    
    if chance:
        labelsColPl[startLabel] = duplabel
    else:
        labelsRowPl[startLabel] = duplabel

    chance = not chance

    while not np.array_equal(np.setdiff1d(np.union1d(labelsRowPl, labelsColPl), np.array([-1])), np.arange(n1+n2)):

        tableau = tableauQ if chance else tableauP
        nrows = n1 if chance else n2
        basis = basisCol if chance else basisRow

        droplabel = duplabel
        duplabel = integerPivot(duplabel, tableau, nrows, basis)
        # print(duplabel)
        # print(tableauP)
        # print(basisRow)
        
        if chance:
            labelsColPl[droplabel] = duplabel
        else:
            labelsRowPl[droplabel] = duplabel

        chance = not chance
        # print(duplabel)

    outputRow = []
    for i in range(n1):
        outputRow.append(0)

    tsum = 0
    for i in range(n2):
        if basisRow[i] < n1:
            outputRow[basisRow[i]] = tableauP[i,-1] / tableauP[i,basisRow[i]]
            tsum += outputRow[basisRow[i]]
    
    for i in range(n1):
        outputRow[i] /= tsum

    outputCol = []
    for i in range(n2):
        outputCol.append(0)

    tsum = 0    
    for i in range(n1):
        if basisCol[i] >= n1:
            outputCol[basisCol[i]-n1] = tableauQ[i,-1] / tableauQ[i,basisCol[i]]
            tsum += outputCol[basisCol[i]-n1]
    
    for i in range(n2):
        outputCol[i] /= tsum

    # print(outputRow)
    # print(outputCol)
    return tuple(outputRow), tuple(outputCol)

P_copy = np.copy(P)
Q_copy = np.copy(Q)

eqset = set()
for i in range(n1+n2):
    chance = 0 if i < n1 else 1
    row_s, col_s = lemkeHowson(n1, n2, i, P_copy, Q_copy, chance)
    eqset.add((row_s, col_s))
    P_copy = np.copy(P)
    Q_copy = np.copy(Q)

print(len(eqset))
for eq in eqset:
    print(" ".join([str(prob) for prob in eq[0]]))
    print(" ".join([str(prob) for prob in eq[1]]))
    