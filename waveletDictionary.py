import numpy as np
from scipy.sparse import csr_matrix
from auxiliar import reshape

def waveletDictionary(timeAxis, nbFrequencies, gaussianSupport = 0.05, threshold = 0.001):
    """
    This function returns a dictionary of gabor functions as the one that is built in:
    http://web.cvxr.com/cvx/examples/cvxbook/Ch06_approx_fitting/html/basispursuit.html
    nbSinusoids = (2 * nbFrequencies + 1) -> real and imaginary parts plus DC component
    """
    print ("building wavelet dictionary ...")

    dictionary = np.exp(- timeAxis**2 / gaussianSupport**2)

    nbNonZeros = sum(i >= threshold for i in dictionary) - 1

    supportIndexes = np.hstack((np.arange(nbNonZeros, -1, -1), np.arange(1, nbNonZeros + 1)))
    dictionary = dictionary[supportIndexes]

    supportLength = dictionary.size
    timeAxisLength = timeAxis.size

    timeAxisIndexes = np.arange(0, timeAxisLength, dtype = np.int16) #is int16 enough?
    columnIndexes = np.repeat(timeAxisIndexes, supportLength)
    dictionaryValues = np.tile(dictionary, timeAxisLength)
    atomIndexes = np.arange(0, supportLength, dtype = np.int16)
    rowIndexes = np.tile(atomIndexes, timeAxisLength) + columnIndexes

    dictionary = csr_matrix((dictionaryValues, (rowIndexes, columnIndexes)), shape = ((timeAxisLength + supportLength -1), timeAxisLength))
    dictionary = dictionary[nbNonZeros: (nbNonZeros + timeAxisLength), :]

    # Defining a basis of sines and cosines 
    freqFactors = np.arange(1, nbFrequencies + 1)
    freqFactors = np.repeat(freqFactors, 2)
    freqFactors = np.insert(freqFactors, 0, 0)

    phases = np.tile(np.array([0.0, -(0.5 * np.pi)]), nbFrequencies) #to switch between sine and cosine
    phases = np.insert(phases, 0, 0)

    baseFrequecy = 5;     # base * nbWavelets = 150 for good results (is it related with nyquist frequency?)
    nbSinusoids = 2 * nbFrequencies + 1

    sinusoidValues = np.zeros(timeAxisLength * nbSinusoids)
    offset = 0;
    for i in range(nbSinusoids):
        frequency = freqFactors[i] * baseFrequecy
        phase = phases[i]
        for j in range(timeAxisLength):
            sinusoidValues[offset] = np.cos(frequency * timeAxis[j] + phase)
            offset += 1

    rowIndexes = np.arange(0, timeAxisLength * nbSinusoids)
    columnIndexes = np.remainder(rowIndexes, timeAxisLength)

    dictionary = csr_matrix((sinusoidValues, (rowIndexes, columnIndexes)), shape = (offset, timeAxisLength)) * dictionary  
    newShape = (timeAxisLength, offset)
    
    dictionary = reshape(dictionary, newShape)
    print ("dictionary is ready!")
    
    return dictionary
