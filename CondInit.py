from parameters import lambda_th

import sys
print(sys.path)

import os
print(os.path.dirname(__file__))

def CI(N,T,L):
    l_th = lambda_th(T)
    return l_th #temporaire

