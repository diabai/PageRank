import numpy as np
from numpy import matrix as Matrix


def main():
    np.seterr(divide="ignore", invalid="ignore")
    mMatrix = Matrix(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])

    adjacencyMatrix = np.genfromtxt('graph.txt', dtype=np.int64, delimiter=" ")
    print("Adjacency Matrix")
    print(adjacencyMatrix)

    #Finds the largest index in both columns
    largestIndex = adjacencyMatrix[:,0:2].max()
    n = largestIndex+1
    # Matrix of size of file's matrix
    mMatrix = np.zeros((n, n))
    for row in adjacencyMatrix:
        i, j, k= row
        mMatrix[j, i] = k


    print("mMatrix")
    print(mMatrix)
    beta = 0.85
    # Gets column sum
    sumVector = mMatrix.sum(0)
    #print(sumVector)
    #Replace all 0s with 1
    sumVector[sumVector == 0] = 1
    #print(sumVector)

    stochasticArray = mMatrix / sumVector

    #print(stochasticArray)

    # Beta * stochastic array
    betaTimesStoch = stochasticArray * beta
    # print(betaTimesStoch)

     # Size of a column
    # print("Shape", mMatrix.shape[0])
    n = mMatrix.shape[0]
    #Leap probabilty
    leapProp = (1-beta)/n
    #print(leapProp)

    #mPrimeArray = betaTimesStoch #+ leapProp
    # Normalizing mPrimeArray (i.e: Stochastic version of mPrimeArray
   # mPrimeArray = mPrimeArray / mPrimeArray.sum(0)

    #print(mPrimeArray)
    #print(mPrimeArray.sum(0))

    # Repeat creates a list of 1 of size n  ; .transpose turn matrix to vertical
    # Original rank vector
    r = Matrix(np.repeat(np.divide(1,n), n)).transpose()
    print("Original rank vector:")
    print(r)

    # nextR = mPrimeArray * r
    prevR = np.zeros(r.shape)

  # print(prevR)
  #  print(prevR.sum())


    iterations = 0
    #print( mMatrix.sum(0))

    while not np.allclose(r, prevR, rtol= 1.e-3, atol=1.e-3):
        prevR = r
        r = (betaTimesStoch * prevR) + leapProp
        iterations += 1

    #Convereged rank vector
    print("Converged R vector:")
    print(r)

    print("Previous R:")
    print(prevR)

    #Total iterations

    print("Total iterations", iterations)

    print("Stochastic matrix: ")
    print(stochasticArray)


if __name__ == "__main__":
    main()

