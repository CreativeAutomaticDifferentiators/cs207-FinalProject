# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:52:34 2019

@author: pelan
"""
import numpy as np

class Var:
    ''' TODO: 
        
        
    '''
    def __init__(self, value, variables):
        self.value = value #value of the expression
        self.variables = variables #list of the variable that intervene in the expression and from which we will be calculating the partial derivatives
        self.grad = {} #dictionary with the partial derivatives values for each variables
        for var in variables:
            self.grad[var]=1
        
    def __radd__(self, other):
        return self + other
    
    def __add__(self, other):
        if(type(other)==int or type(other)==float):
            z=Var(self.value + other, self.variables)
            z.grad = self.grad
        else:
            z=Var(self.value + other.value, self.variables + other.variables)
            for key in z.grad:
                z.grad[key]=0
                if(key in self.grad):
                    z.grad[key] += self.grad[key]
                if(key in other.grad):
                    z.grad[key] += other.grad[key]
        return z
    
    def __rmul__(self, other):
        return self * other
    
    def __mul__(self, other):
        if(type(other)==int or type(other)==float):
            z=Var(self.value * other, self.variables)
            for key in z.grad:
                z.grad[key] = other * self.grad[key]
        else:
            z=Var(self.value * other.value, self.variables + other.variables)
            for key in z.grad:
                z.grad[key]=0
                if(key in self.grad):
                    z.grad[key] += self.grad[key] * other.value
                if(key in other.grad):
                    z.grad[key] += other.grad[key] * self.value
        return z
    


#define your own function            
def as_function(f, f_derivative):
    '''TODO: properly!!!!!!!!!!!
    
    f: the function
    f_derivative: its derivative
    
    g is a function that transforms a type Var expression by applying:
        f to Var.value
        and the chain rule with f_derivative to its gradient
    g returns a new type Var object.
    '''
    
    def g(var):
        if(type(var)==int or type(var)==float):
            return f(var)
        else:
            z=Var(f(var.value), var.variables)
            for key in var.grad:
                z.grad[key] = f_derivative(var.value)*var.grad[key]
            return z
    return g



#Let's define some basics function
sin = as_function (np.sin, np.cos)

def cos_p(x):
    return -np.sin(x)
cos = as_function(np.cos, cos_p)

exp = as_function(np.exp, np.exp)

def log_p(x):
    return 1/x
ln = as_function(np.log, log_p)
