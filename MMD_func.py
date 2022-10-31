#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:48:46 2022

@author: nokicheng
"""

import numpy as np

def norm_matrix(matrix_1, matrix_2):
    norm_square_1 = np.sum(np.square(matrix_1), axis = 1)
    norm_square_1 = np.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = np.sum(np.square(matrix_2), axis = 1)
    norm_square_2 = np.reshape(norm_square_2, (-1,1))
    
    inner_matrix = np.matmul(matrix_1, np.transpose(matrix_2))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + np.transpose(norm_square_2)
    
    return np.maximum(0, norm_diff)
    
def inner_matrix(matrix_1, matrix_2):
    return np.matmul(matrix_1, np.transpose(matrix_2))

def kernel_RBF(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    K =  np.exp(-matrix/ (sigma**2))
    return K

def MMD(P,Q,para):
    pm,n1 = P.shape
    qm,n2 = Q.shape
    if n1 != n2:
        print('Data Dimension not match')
        return
    KPP = kernel_RBF(P,P,para)
    KPQ = kernel_RBF(P,Q,para)
    KQQ = kernel_RBF(Q,Q,para)
    mmd = (KPP.sum()-np.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-np.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
    return mmd