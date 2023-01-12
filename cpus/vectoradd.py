import sys, os, glob, shutil
import numpy as np
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)



def vectorAdd(a,b,c):
    for i in range(a.size):
        c[i]=a[i]+b[i]
    

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

    vectorAdd(A,B,C)
    
    vectoradd_time=timer()-start
    print("VectorAdd took %f seconds"%(vectoradd_time) )

    start = timer()
    C=A+B
    vectoradd_time=timer()-start
    print("Usual add took %f seconds"%(vectoradd_time) )
    
    
    
if __name__ == "__main__":
    main()
    

