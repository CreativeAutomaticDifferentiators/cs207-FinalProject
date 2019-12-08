import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from forward import *
import numpy as np

def test_add_objects():
    x = Variable(1, np.array([1,0,0]))
    y = Variable(2, np.array([0,1,0]))
    z = Variable(3, np.array([0,0,1]))
    f = x + y + z
    assert f.val == 6, "Error in add: incorrect value"
    assert (f.der == np.array([1,1,1])).all(), "Error in add: incorrect derivative"

def test_mul():
    x = Variable(3)
    y = x * 2
    assert y.val == 6, "Error in mul: incorrect value"
    assert y.der == 2, "Error in mul: incorrect derivative"

def test_mul_neg():
    x = Variable(3)
    y = x * -4
    assert y.val == -12, "Error in mul: incorrect value"
    assert y.der == -4, "Error in mul: incorrect derivative"

def test_rmul():
    x = Variable(3)
    y = 2 * x
    assert y.val == 6, "Error in rmul: incorrect value"
    assert y.der == 2, "Error in rmul: incorrect derivative"

def test_rmul_neg():
    x = Variable(3)
    y = -4 * x
    assert y.val == -12, "Error in rmul: incorrect value"
    assert y.der == -4, "Error in rmul: incorrect derivative"

def test_add():
    x = Variable(8)
    y = x + 3
    assert y.val == 11, "Error in add: incorrect value"
    assert y.der == 1, "Error in add: incorrect derivative"

def test_add_neg():
    x = Variable(8)
    y = x + (-3)
    assert y.val == 5, "Error in add: incorrect value"
    assert y.der == 1, "Error in add: incorrect derivative"

def test_radd():
    x = Variable(8)
    y = 3 + x
    assert y.val == 11, "Error in radd: incorrect value"
    assert y.der == 1, "Error in radd: incorrect derivative"

def test_radd_neg():
    x = Variable(8)
    y = -3 + x
    assert y.val == 5, "Error in radd: incorrect value"
    assert y.der == 1, "Error in radd: incorrect derivative"

def test_sub():
    x = Variable(8)
    y = x - 3
    assert y.val == 5, "Error in sub: incorrect value"
    assert y.der == 1, "Error in sub: incorrect derivative"

def test_rsub():
    x = Variable(8)
    y = 3 - x
    assert y.val == -5, "Error in rsub: incorrect value"
    assert y.der == -1, "Error in rsub: incorrect derivative"

def test_sin():
    x = Variable(np.pi)
    y = np.sin(x)
    assert y.val == np.sin(np.pi), "Error in sin: incorrect value"
    assert y.der == np.cos(np.pi), "Error in sin: incorrect derivative"

def test_sin_neg():
    x = Variable(np.pi)
    y = np.sin(-x)
    assert y.val == np.sin(-np.pi), "Error in sin: incorrect value"
    assert y.der == -np.cos(-np.pi), "Error in sin: incorrect derivative"

def test_cos():
    x = Variable(np.pi)
    y = np.cos(x)
    assert y.val == np.cos(np.pi), "Error in cos: incorrect value"
    assert y.der == -np.sin(np.pi), "Error in cos: incorrect derivative"

def test_cos_neg():
    x = Variable(np.pi)
    y = np.cos(-x)
    assert y.val == np.cos(-np.pi), "Error in cos: incorrect value"
    assert y.der == np.sin(-np.pi), "Error in cos: incorrect derivative"

def test_exp():
    x = Variable(3)
    y = np.exp(x)
    assert y.val == np.exp(3), "Error in exp: incorrect value"
    assert y.der == np.exp(3), "Error in exp: incorrect derivative"

def test_exp_neg():
    x = Variable(3)
    y = np.exp(-x)
    assert y.val == np.exp(-3), "Error in exp: incorrect value"
    assert y.der == -np.exp(-3), "Error in exp: incorrect derivative"

def test_log():
    x = Variable(4)
    y = np.log(x)
    assert y.val == np.log(4), "Error in log: incorrect value"
    assert y.der == 1/4, "Error in log: incorrect derivative"

def test_log_neg():
    x = Variable(-4)
    y = np.log(-x)
    assert y.val == np.log(4), "Error in log: incorrect value"
    assert y.der == -1/4, "Error in log: incorrect derivative"

def test_tan():
    x = Variable(4)
    y = np.tan(x)
    assert y.val == np.tan(4), "Error in tan: incorrect value"
    assert y.der == 1/np.cos(4)**2, "Error in tan: incorrect derivative"

def test_pow():
    x = Variable(3)
    y = x**4
    assert y.val == 81, "Error in pow: incorrect value"
    assert y.der == 108, "Error in pow: incorrect derivative"

def test_pow_neg():
    x = Variable(3)
    y = x**(-2)
    assert y.val == 1/9, "Error in pow: incorrect value"
    assert y.der == -2/27, "Error in pow: incorrect derivative"

def test_rpow():
    x = Variable(3)
    y = (2)**x
    assert y.val == 8, "Error in rpow: incorrect value"
    assert y.der == 8*np.log(2), "Error in rpow: incorrect derivative"

def test_sqrt():
    x = Variable(4)
    y = np.sqrt(x)
    assert y.val == np.sqrt(4), "Error in sqrt: incorrect value"
    assert y.der == 0.5/np.sqrt(4), "Error in sqrt: incorrect derivative"

def test_truediv():
    x = Variable(10)
    y = x/4
    assert y.val == 10/4, "Error in truediv: incorrect value"
    assert y.der == 1/4, "Error in truediv: incorrect derivative"

def test_rtruediv():
    x = Variable(10)
    y = 4/x
    assert y.val == 4/10, "Error in rtruediv: incorrect value"
    assert y.der == -4/100, "Error in rtruediv: incorrect derivative"

def test_neg():
    x = Variable(10)
    y = -x
    assert y.val == -10, "Error in neg: incorrect value"
    assert y.der == -1, "Error in neg: incorrect derivative"

def test_jacobian():
    x = Variable(10)
    y = -x
    assert y.der == y.get_jacobian(), "Error in get_jacobian: incorrect value"

def test_arctan():
    x = Variable(5)
    y = np.arctan(x)
    assert y.val == np.arctan(5), "Error in arctan: incorrect value"
    assert y.der == 1/(1+25), "Error in arctan: incorrect derivative"

def test_arccos():
    x = Variable(0.5)
    y = np.arccos(x)
    assert y.val == np.arccos(0.5), "Error in arccos: incorrect value"
    assert y.der == -1/np.sqrt(1-0.25), "Error in arccos: incorrect derivative"

def test_arcsin():
    x = Variable(0.5)
    y = np.arcsin(x)
    assert y.val == np.arcsin(0.5), "Error in arcsin: incorrect value"
    assert y.der == 1/np.sqrt(1-0.25), "Error in arcsin: incorrect derivative"

def test_tan_invalid():
    with pytest.raises(ValueError):
        x = Variable(np.pi/2)
        y = np.tan(x)

def test_arccos_invalid():
    with pytest.raises(ValueError):
        x = Variable(5)
        y = np.arccos(x)

def test_arcsin_invalid():
    with pytest.raises(ValueError):
        x = Variable(5)
        y = np.arcsin(x)

def test_truediv_invalid_zero():
    with pytest.raises(ZeroDivisionError):
        x = Variable(5)
        y = x/0

def test_rtruediv_invalid_zero():
    with pytest.raises(ZeroDivisionError):
        x = Variable(0)
        y = 2/x

def test_add_invalid():
    with pytest.raises(TypeError):
        x = Variable(5)
        y = x + 'string'

def test_mul_invalid():
    with pytest.raises(TypeError):
        x = Variable(5)
        y = x * 'string'

def test_truediv_invalid_type():
    with pytest.raises(TypeError):
        x = Variable(5)
        y = x / 'string'

def test_rtruediv_invalid_type():
    with pytest.raises(TypeError):
        x = Variable(5)
        y = 'string' / x

def test_mul_objects():
    x = Variable(5, np.array([1,0]))
    y = Variable(4, np.array([0,1]))
    z = x * y
    assert z.val == 20, "Error in mul: incorrect value"
    assert (z.der == np.array([4,5])).all(), "Error in mul: incorrect derivative"
    
def test_truediv_objects():
    x = Variable(4, np.array([1,0]))
    y = Variable(2, np.array([0,1]))
    z = x / y
    assert z.val == 2, "Error in truediv: incorrect value"
    assert (z.der == np.array([1/2,-1])).all(), "Error in truediv: incorrect derivative"

def test_rtruediv_objects():
    x = Variable(4, np.array([1,0]))
    y = Variable(2, np.array([0,1]))
    z = y / x
    assert z.val == 1/2, "Error in rtruediv: incorrect value"
    assert (z.der == np.array([-2/16,1/4])).all(), "Error in rtruediv: incorrect derivative"
