# Implements the method described in:
#@article{park_robot_1994,
#  title = {Robot sensor calibration: solving {AX}= {XB} on the {Euclidean} group},
#  volume = {10},
#  number = {5},
#  journal = {IEEE Transactions on Robotics and Automation},
#  author = {Park, Frank C and Martin, Bryan J},
#  year = {1994},
#  pages = {717--721}
#}

import numpy
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv

def log(R):
    # Rotation matrix logarithm
    theta = numpy.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    return numpy.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*numpy.sin(theta))

def invsqrt(mat):
    u,s,v = numpy.linalg.svd(mat)
    return u.dot(numpy.diag(1.0/numpy.sqrt(s))).dot(v)

def calibrate(A, B):
    #transform pairs A_i, B_i
    N = len(A)
    M = numpy.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    return Rx, tx.flatten()

