from forward import *
import pytest

def test_addition():
    x = Variable(1, np.array([1,0,0]))
    y = Variable(2, np.array([0,1,0]))
    z = Variable(3, np.array([0,0,1]))

    f = x + y + z

    assert f.val == 6, "Error in Addition: value does not match correct value"
    assert (f.der == np.array([1,1,1])).all(), "Error in Addition: derivative does not match correct value"

# if __name__ == "__main__":
#     test_addition()