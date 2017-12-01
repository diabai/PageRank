import numpy as np
from numpy import matrix as Matrix

# Author: Ibrahim Diabate

def main():
    np.seterr(divide="ignore", invalid="ignore")
    adjacencyMatrix = np.genfromtxt('graph.txt', dtype=np.int64, delimiter=" ")

    print("Adjacency Matrix from file")
    print(adjacencyMatrix)

    #Finds the largest index in both columns
    largestIndex = adjacencyMatrix[:,0:2].max()
    n = largestIndex+1
    # Creates matrix of size of file's matrix
    mMatrix = np.zeros((n, n))
    for row in adjacencyMatrix:
        i, j, k= row
        mMatrix[j, i] = k

    print("Connection Matrix")
    print(mMatrix)
    beta = 0.85
    # Gets columns sums
    sumVector = mMatrix.sum(0)
    #Replace all 0s with 1
    sumVector[sumVector == 0] = 1

    # Generates the initial stochastic matrix
    stochasticArray = mMatrix / sumVector

    print("Stochastic matrix: ")
    print(stochasticArray)

    # Beta * stochastic array
    betaTimesStoch = stochasticArray * beta

    # Size of a column
    n = mMatrix.shape[0]
    #Leap probabilty
    leapProp = (1-beta)/n

    # Repeat creates a list of 1 of size n; .transpose turn matrix to vertical
    # Original rank vector
    r = Matrix(np.repeat(np.divide(1,n), n)).transpose()
    print("Original rank vector:")
    print(r)

    prevR = np.zeros(r.shape)

    iterations = 0
    # Looping until they converge
    while not np.allclose(r, prevR, rtol= 1.e-3, atol=1.e-3):
        prevR = r
        r = (betaTimesStoch * prevR) + leapProp
        iterations += 1

    #Convereged rank vector
    print("Converged Rank vector:")
    print(r)

    #print("Previous R:")
   # print(prevR)

    #Total iterations

    print("Total iterations", iterations)


if __name__ == "__main__":
    main()

