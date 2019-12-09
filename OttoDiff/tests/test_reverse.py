import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from reverse import *
import numpy as np

def test_add_objects():
    x = VariableNode(1, np.array([1,0,0]))
    y = VariableNode(2, np.array([0,1,0]))
    z = VariableNode(3, np.array([0,0,1]))
    f = x + y + z
    assert f.val == 6, "Error in add: incorrect value"
    assert find_df_dx(f=f, x=x) == 1, "Error in add: incorrect derivative"
    assert find_df_dx(f=f, x=y) == 1, "Error in add: incorrect derivative"
    assert find_df_dx(f=f, x=z) == 1, "Error in add: incorrect derivative"

def test_mul():
    x = VariableNode(3)
    y = x * 2
    assert y.val == 6, "Error in mul: incorrect value"
    assert find_df_dx(f=y, x=x) == 2, "Error in mul: incorrect derivative"

def test_mul_neg():
    x = VariableNode(3)
    y = x * -4
    assert y.val == -12, "Error in mul: incorrect value"
    assert find_df_dx(f=y, x=x) == -4, "Error in mul: incorrect derivative"

def test_rmul():
    x = VariableNode(3)
    y = 2 * x
    assert y.val == 6, "Error in rmul: incorrect value"
    assert find_df_dx(f=y, x=x) == 2, "Error in rmul: incorrect derivative"

def test_rmul_neg():
    x = VariableNode(3)
    y = -4 * x
    assert y.val == -12, "Error in rmul: incorrect value"
    assert find_df_dx(f=y, x=x) == -4, "Error in rmul: incorrect derivative"

def test_add():
    x = VariableNode(8)
    y = x + 3
    assert y.val == 11, "Error in add: incorrect value"
    assert find_df_dx(f=y, x=x) == 1, "Error in add: incorrect derivative"

def test_add_neg():
    x = VariableNode(8)
    y = x + (-3)
    assert y.val == 5, "Error in add: incorrect value"
    assert find_df_dx(f=y, x=x) == 1, "Error in add: incorrect derivative"

def test_radd():
    x = VariableNode(8)
    y = 3 + x
    assert y.val == 11, "Error in radd: incorrect value"
    assert find_df_dx(f=y, x=x) == 1, "Error in radd: incorrect derivative"

def test_radd_neg():
    x = VariableNode(8)
    y = -3 + x
    assert y.val == 5, "Error in radd: incorrect value"
    assert find_df_dx(f=y, x=x) == 1, "Error in radd: incorrect derivative"

def test_sub():
    x = VariableNode(8)
    y = x - 3
    assert y.val == 5, "Error in sub: incorrect value"
    assert find_df_dx(f=y, x=x) == 1, "Error in sub: incorrect derivative"

def test_rsub():
    x = VariableNode(8)
    y = 3 - x
    assert y.val == -5, "Error in rsub: incorrect value"
    assert find_df_dx(f=y, x=x) == -1, "Error in rsub: incorrect derivative"

def test_sin():
    x = VariableNode(np.pi)
    y = np.sin(x)
    assert y.val == np.sin(np.pi), "Error in sin: incorrect value"
    assert find_df_dx(f=y, x=x) == np.cos(np.pi), "Error in sin: incorrect derivative"

def test_sin_neg():
    x = VariableNode(np.pi)
    y = np.sin(-x)
    assert y.val == np.sin(-np.pi), "Error in sin: incorrect value"
    assert find_df_dx(f=y, x=x) == -np.cos(-np.pi), "Error in sin: incorrect derivative"

def test_cos():
    x = VariableNode(np.pi)
    y = np.cos(x)
    assert y.val == np.cos(np.pi), "Error in cos: incorrect value"
    assert find_df_dx(f=y, x=x) == -np.sin(np.pi), "Error in cos: incorrect derivative"

def test_cos_neg():
    x = VariableNode(np.pi)
    y = np.cos(-x)
    assert y.val == np.cos(-np.pi), "Error in cos: incorrect value"
    assert find_df_dx(f=y, x=x) == np.sin(-np.pi), "Error in cos: incorrect derivative"

def test_exp():
    x = VariableNode(3)
    y = np.exp(x)
    assert y.val == np.exp(3), "Error in exp: incorrect value"
    assert find_df_dx(f=y, x=x) == np.exp(3), "Error in exp: incorrect derivative"

def test_exp_neg():
    x = VariableNode(3)
    y = np.exp(-x)
    assert y.val == np.exp(-3), "Error in exp: incorrect value"
    assert find_df_dx(f=y, x=x) == -np.exp(-3), "Error in exp: incorrect derivative"

def test_log():
    x = VariableNode(4)
    y = np.log(x)
    assert y.val == np.log(4), "Error in log: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/4, "Error in log: incorrect derivative"

def test_log_neg():
    x = VariableNode(-4)
    y = np.log(-x)
    assert y.val == np.log(4), "Error in log: incorrect value"
    assert find_df_dx(f=y, x=x) == -1/4, "Error in log: incorrect derivative"

def test_tan():
    x = VariableNode(4)
    y = np.tan(x)
    assert y.val == np.tan(4), "Error in tan: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/np.cos(4)**2, "Error in tan: incorrect derivative"

def test_pow():
    x = VariableNode(3)
    y = x**4
    assert y.val == 81, "Error in pow: incorrect value"
    assert find_df_dx(f=y, x=x) == 108, "Error in pow: incorrect derivative"

def test_pow_neg():
    x = VariableNode(3)
    y = x**(-2)
    assert y.val == 1/9, "Error in pow: incorrect value"
    assert find_df_dx(f=y, x=x) == -2/27, "Error in pow: incorrect derivative"

def test_rpow():
    x = VariableNode(3)
    y = (2)**x
    assert y.val == 8, "Error in rpow: incorrect value"
    assert find_df_dx(f=y, x=x) == 8*np.log(2), "Error in rpow: incorrect derivative"

def test_sqrt():
    x = VariableNode(4)
    y = np.sqrt(x)
    assert y.val == np.sqrt(4), "Error in sqrt: incorrect value"
    assert find_df_dx(f=y, x=x) == 0.5/np.sqrt(4), "Error in sqrt: incorrect derivative"

def test_truediv():
    x = VariableNode(10)
    y = x/4
    assert y.val == 10/4, "Error in truediv: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/4, "Error in truediv: incorrect derivative"

def test_rtruediv():
    x = VariableNode(10)
    y = 4/x
    assert y.val == 4/10, "Error in rtruediv: incorrect value"
    assert find_df_dx(f=y, x=x) == -4/100, "Error in rtruediv: incorrect derivative"

def test_neg():
    x = VariableNode(10)
    y = -x
    assert y.val == -10, "Error in neg: incorrect value"
    assert find_df_dx(f=y, x=x) == -1, "Error in neg: incorrect derivative"

def test_arctan():
    x = VariableNode(5)
    y = np.arctan(x)
    assert y.val == np.arctan(5), "Error in arctan: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/(1+25), "Error in arctan: incorrect derivative"

def test_arccos():
    x = VariableNode(0.5)
    y = np.arccos(x)
    assert y.val == np.arccos(0.5), "Error in arccos: incorrect value"
    assert find_df_dx(f=y, x=x) == -1/np.sqrt(1-0.25), "Error in arccos: incorrect derivative"

def test_arcsin():
    x = VariableNode(0.5)
    y = np.arcsin(x)
    assert y.val == np.arcsin(0.5), "Error in arcsin: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/np.sqrt(1-0.25), "Error in arcsin: incorrect derivative"

def test_tan_invalid():
    with pytest.raises(ValueError):
        x = VariableNode(np.pi/2)
        y = np.tan(x)

def test_arccos_invalid():
    with pytest.raises(ValueError):
        x = VariableNode(5)
        y = np.arccos(x)

def test_arcsin_invalid():
    with pytest.raises(ValueError):
        x = VariableNode(5)
        y = np.arcsin(x)

def test_truediv_invalid_zero():
    with pytest.raises(ZeroDivisionError):
        x = VariableNode(5)
        y = x/0

def test_rtruediv_invalid_zero():
    with pytest.raises(ZeroDivisionError):
        x = VariableNode(0)
        y = 2/x

def test_add_invalid():
    with pytest.raises(TypeError):
        x = VariableNode(5)
        y = x + 'string'

def test_mul_invalid():
    with pytest.raises(TypeError):
        x = VariableNode(5)
        y = x * 'string'

def test_truediv_invalid_type():
    with pytest.raises(TypeError):
        x = VariableNode(5)
        y = x / 'string'

def test_rtruediv_invalid_type():
    with pytest.raises(TypeError):
        x = VariableNode(5)
        y = 'string' / x

def test_mul_objects():
    x = VariableNode(5, np.array([1,0]))
    y = VariableNode(4, np.array([0,1]))
    z = x * y
    assert z.val == 20, "Error in mul: incorrect value"
    assert find_df_dx(f=z, x=x) == 4, "Error in mul: incorrect derivative"
    assert find_df_dx(f=z, x=y) == 5, "Error in mul: incorrect derivative"
    
def test_truediv_objects():
    x = VariableNode(4, np.array([1,0]))
    y = VariableNode(2, np.array([0,1]))
    z = x / y
    assert z.val == 2, "Error in truediv: incorrect value"
    assert find_df_dx(f=z, x=x) == 1/2, "Error in truediv: incorrect derivative"
    assert find_df_dx(f=z, x=y) == -1, "Error in truediv: incorrect derivative"

def test_rtruediv_objects():
    x = VariableNode(4, np.array([1,0]))
    y = VariableNode(2, np.array([0,1]))
    z = y / x
    assert z.val == 1/2, "Error in rtruediv: incorrect value"
    assert find_df_dx(f=z, x=x) == -2/16, "Error in rtruediv: incorrect derivative"
    assert find_df_dx(f=z, x=y) == 1/4, "Error in rtruediv: incorrect derivative"

def test_logarithm():
    x = VariableNode(8)
    y = x.logarithm(2)
    assert y.val == 3, "Error in logarithm: incorrect value"
    assert find_df_dx(f=y, x=x) == 1/(8*np.log(2)), "Error in logarithm: incorrect derivative"

def test_logarithm_neg():
    x = VariableNode(27)
    with pytest.raises(ValueError):
        y = x.logarithm(-3)

def test_logarithm_one():
    x = VariableNode(27)
    with pytest.raises(ValueError):
        y = x.logarithm(1)

def test_logistic():
    x = VariableNode(4)
    y = x.logistic()
    gx = 1/(1+np.exp(-4))
    assert y.val == gx, "Error in logistic: incorrect value"
    assert find_df_dx(f=y, x=x) == gx*(1-gx), "Error in logistic: incorrect derivative"

def test_logistic_neg():
    x = VariableNode(-4)
    y = x.logistic()
    gx = 1/(1+np.exp(4))
    assert y.val == gx, "Error in logistic: incorrect value"
    assert find_df_dx(f=y, x=x) == gx*(1-gx), "Error in logistic: incorrect derivative"

def test_sinh():
    x = VariableNode(5)
    y = x.sinh()
    assert y.val == (np.exp(5)-np.exp(-5))/2, "Error in sinh: incorrect value"
    assert find_df_dx(f=y, x=x) == np.cosh(5), "Error in sinh: incorrect derivative"

def test_cosh():
    x = VariableNode(5)
    y = x.cosh()
    assert y.val == (np.exp(5)+np.exp(-5))/2, "Error in cosh: incorrect value"
    assert find_df_dx(f=y, x=x) == np.sinh(5), "Error in cosh: incorrect derivative"

def test_tanh():
    x = VariableNode(5)
    y = x.tanh()
    assert y.val == (np.exp(5)-np.exp(-5))/(np.exp(5)+np.exp(-5)), "Error in tanh: incorrect value"
    assert find_df_dx(f=y, x=x) == (1/np.cosh(5))**2, "Error in tanh: incorrect derivative"
