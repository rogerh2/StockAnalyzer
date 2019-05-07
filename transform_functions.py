import numpy as np

def unity(t):
    return t

def inv_t(t):
    ans = 1 / t
    return ans

def inv_sq_t(t):
    ans = -inv_t(t**2)
    return ans

def exp_decay(t):
    ans = np.exp(-t)
    return ans


functions = {'unity':unity, 'inv_t':inv_t, 'inv_sq_t':inv_sq_t, 'exp_decay':exp_decay}