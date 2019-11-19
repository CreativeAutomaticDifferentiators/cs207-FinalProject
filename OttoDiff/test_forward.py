import pytest
from forward import *
import numpy as np

def test_addition():
    x = Variable(1, np.array([1,0,0]))
    y = Variable(2, np.array([0,1,0]))
    z = Variable(3, np.array([0,0,1]))
    
    f = x + y + z
    
    assert f.val == 6, "Error in Addition: value does not match correct value"
    assert (f.der == np.array([1,1,1])).all(), "Error in Addition: derivative does not match correct value"


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
