import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

N = 1000 # number of datapoints
C = 1

start = numpy.zeros(N)
bounds=[(0, C) for b in range(N)]

def linear_kernel(x,y):
    return numpy.dot(x,y)

def polynomial_kernel(x,y,p):
    return (numpy.dot(x,y)+1)**p

def RBF(x,y,sigma):
    return numpy.exp(-((numpy.linalg.norm(numpy.array(x)-numpy.array(y)))**2)/(2*sigma**2))

def sub_obj(t,x):
    P = numpy.array([  numpy.array([ t[i]*t[j]*linear_kernel(x[i],x[j]) for j in range(len(x)) ]) for i in range(len(x)) ])
    return P

# P = sub_obj(t,x)

def objective(alpha):
    s=0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            s+=alpha[i]*alpha[j]*sub_obj(t,x)[i][j]
    return s

    return numpy.dot([ numpy.dot(sub_obj(t,x)[i,:],alpha) for i in range(len(alpha)) ], alpha)

def zerofun(alpha,t):
    return numpy.dot(alpha,t)

constraint={'type':'eq', 'fun':zerofun}

ret = minimize (objective,start,bounds=bounds,constraints=constraint)
#alpha=ret['x']

values = []
thr=1e-5
for i,num in enumerate(alpha):
    if num>thr:
        values.append([num,x[i],t[i]])

s=values[0][1]
ts=values[0][2]

b=0
for i in range(N):
    b+=alpha[i]*t[i]*linear_kernel(s,x[i])
b-=ts

def indicator(s):
    ind=0
    for i in range(len(values)):
        ind+=values[i][0]*values[i][2]*linear_kernel(s,values[i][1])
    ind-=b
    return ind

