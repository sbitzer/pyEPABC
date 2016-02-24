import numpy as np
cimport numpy as np
import datetime

cdef extern from "DDMsampler.h":
    int sample_from_DDM_C(double *parsamples, int N, int P, double dt, 
                          double maxRT, int *seed, double *choices, double *RTs)

def sample_from_DDM(np.ndarray[np.float_t, ndim=2, mode="c"] parsamples not None, 
                    int dind, np.ndarray[np.int_t, ndim=1] truetarget, 
                    double dt=0.01, double maxRT=2.5):
    """
    sample_from_DDM(parsamples, dind, truetarget, dt=0.01, maxRT=2.5)
    
    Cython interface to the pure C function.
    
    It makes the output ndarray and hands over the appropriate pointers to the
    C function. Argument dind is not needed in the C function, because I here 
    flip the drift according to which option was the true option in the 
    stimulus.
    
    This module needs to be compiled before it can be used, see README.
    """
    cdef int N = parsamples.shape[0]
    cdef int P = parsamples.shape[1]
    cdef int D = 2
    
    # every time a new seed for the random number generator is used
    # Note, though, that the value generated here will be overwritten during 
    # runtime of this function call
    cdef int seed = np.random.randint(1000000)
    
    cdef np.ndarray[np.float_t, ndim=2] responses = np.zeros((2, N), 
        dtype=float, order='C')
        
    # if second alternative, flip mean drift from positive to negative
    if truetarget[dind] == 2:
        parsamples[:, 0] = -parsamples[:, 0]
        
    err = sample_from_DDM_C(&parsamples[0, 0], N, P, dt, maxRT, &seed, 
                            &responses[0, 0], &responses[1, 0])
    if err:
        raise RuntimeError('sample_from_DDM_C returned with error code %d!' 
                            % err)
        
    return responses.T