import sys, os, glob, shutil
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)

import numpy as np
from numba import jit, njit, vectorize, cuda

#@vectorize(['float32(float32, float32)'], target='gpu')
@vectorize(['float32(float32, float32)'], target='cuda')
def cudaAdd(a,b):
    return a + b

def vectorAdd(a,b):
    return a + b
    

def main():    
    #level = logging.DEBUG
    level = logging.INFO
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    logging.basicConfig(format=loggerformat, level=level)

    N=320000000
    A=np.ones(N, dtype=np.float32)
    B=np.ones(N, dtype=np.float32)
    C=np.ones(N, dtype=np.float32)
        
    start = timer()
    numba_add=jit(vectorAdd)
    C=numba_add(A,B)
    vectoradd_time=timer()-start
    print("Numba Add took %f seconds"%(vectoradd_time) )

    start = timer()
    numba_add=jit(vectorAdd)
    C=numba_add(A,B)
    vectoradd_time=timer()-start
    print("Numba Add compiled took %f seconds"%(vectoradd_time) )

    start = timer()
    C=cudaAdd(A,B)
    vectoradd_time=timer()-start
    print("Numba cuda Add took %f seconds"%(vectoradd_time) )

    start = timer()
    C=A+B
    vectoradd_time=timer()-start
    print("Usual add took %f seconds"%(vectoradd_time) )
    
    
    
if __name__ == "__main__":
    main()
    

