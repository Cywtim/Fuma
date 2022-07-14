# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:42:54 2021

@author: CHENG
"""

import xlrd
import numpy as np

def read_excel(filename):
    
    
    data = xlrd.open_workbook(filename)
    
    table = data.sheets()[0]
    
    nrows = table.nrows
    ncols = table.ncols
    
    output = np.zeros((nrows,ncols))
    
    for i in range(ncols):
        cols = table.col_values(i)
        output[:,i] = cols
    
    return output


