import numpy as np

def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newdata = dataMat - meanVal
    return newdata, meanVal



def pca(dataMat, n):
    newdata, meanVal = zeroMean(dataMat)
    covMat = np.cov(newdata, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)

    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:,n_eigValIndice]

    lowDData = newdata * n_eigVect
    reconMat = (lowDData * n_eigVect.T) + meanVal

    return lowDData, reconMat


