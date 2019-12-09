# The design choice is inspired and influenced by the following reference:
# http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf
# https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/
# https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

import numpy as np

def find_df_dx(f, x):
    """Find the partial derivative of f with respect to given x

    EXAMPLES
    =========
    >>> x = VariableNode(3)
    >>> f = 2 * x
    >>> print(find_df_dx(f=v, x=x))
    6
    """
    # set the gradient of final output node to be 1
    f._grad = 1
    # the gradient you you reho
    return x.grad

class VariableNode:
    """Node in ComputationalGraph store intermediate result"""

    # A node ID counter
    idCounter = 0

    def __init__(self, val, parents = [], op=""):
        self.val = val
        # ∂f/∂(self)
        self._grad = None
        # list of tuple (∂(child)/∂(self), child)
        self.children = []
        # use for back traverse the tree
        self.parents = parents
        # all nodes are named X_i
        VariableNode.idCounter += 1
        # the operation and the parents of the node forms an computation
        self.operation = op
        self.name = f"X_{VariableNode.idCounter} [" + '%.3f'%(val) + "] " + self.operation

    @property
    def grad(self):
        if self._grad is None:
            self.compute_grad()
        return self._grad

    def compute_grad(self):
        # by chain rule
        # ∂f/∂(self) = sum (∂(child)/∂(self) * ∂f/∂(child))
        self._grad = sum(p_der * var.grad for p_der, var in self.children)
        return self._grad

    def __str__(self):
        return self.name

    def __neg__(self):
        z = VariableNode(val=-self.val,
                         parents = [self],
                         op = "<neg>")
        # ∂(z) /∂(self)
        # z = -x  -> dz = -1
        self.children.append((-1, z))
        return z

    def __add__(self, other):
        # other is a number
        if isinstance(other, (int, float)):
            other = Constant(other)
        try:
            # create child
            z = VariableNode(val=self.val + other.val,
                             parents=[self, other],
                             op = "<add>")
            # z = x + other  -> dz/dself = 1, dz/dother = 1
            self.children.append((1, z))
            other.children.append((1, z))
            return z
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other)
        try:
            # create child
            z = VariableNode(self.val + other.val,
                             parents = [self, other],
                             op = "<sub>")
            # z = x - other  -> dz/dself = 1, dz/dother = -1
            self.children.append((1, z))
            other.children.append((-1, z))
            return z
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other)
        try:
            # create child
            z = VariableNode(self.val + other.val,
                             parents = [self, other],
                             op = "<sub>")
            # z = other - x  -> dz/dself = -1, dz/dother = 1
            self.children.append((-1, z))
            other.children.append((1, z))
            return z
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other)
        try:
            # create child
            z = VariableNode(self.val * other.val,
                             parents = [self, other],
                             op = "<mul>")
            # z =  x * other  -> dz/dself = other, dz/dother = self
            self.children.append((other.val, z))
            other.children.append((self.val, z))
            return z
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            if isinstance(other, (int, float)):
                other = Constant(other)
            z = VariableNode(self.val / other.val,
                             parents = [self, other],
                             op = "<div>")
            # z = x / other -> dz/dself = 1/other, dz/dother = -x * (other^-2)
            self.children.append((1 / other.val, z))
            other.children.append((- self.val * (other.val ** (-2)), z))
            return z
        except ZeroDivisionError:
            raise ZeroDivisionError("Failed when dividing by 0")
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __rtruediv__(self, other):
        try:
            if isinstance(other, (int, float)):
                other = Constant(other)
            z = VariableNode(other.val / self.val,
                             parents = [self, other],
                             op = "<rdiv>")
            # z = other / x -> dz/dother = 1/x, dz/dself = -other * (x^-2), 1/x
            self.children.append((-other.val * (self.val ** (-2)), z))
            other.children.append((1 / self.val, z))
            return z
        except ZeroDivisionError:
            raise ZeroDivisionError("Failed when dividing by 0")
        except:
            raise TypeError("other is not a valid Variable or number.")

    def __pow__(self, power):
        z = VariableNode(self.val ** power,
                         parents = [self],
                         op = "<pow>")
        # z = x ** p -> dz/dself = p * x^(p-1)
        self.children.append((power * self.val ** (power-1), z))
        return z

    def sqrt(self):
        return self ** 0.5

    def __rpow__(self, base):
        z = VariableNode(base ** self.val,
                         parents = [self],
                         op = "<" + str(base) + "^x>")
        # z = a ** x -> dz/dself = ln(a) * a^f(x)
        self.children.append((np.log(base) * (base ** self.val), z))
        return z

    def exp(self):
        z = VariableNode(np.exp(self.val),
                         parents = [self],
                         op = "<exp>")
        # z = e ** x -> dz/dself = e^f(x)
        self.children.append((np.exp(self.val), z))
        return z

    def log(self):
        z = VariableNode(np.log(self.val),
                         parents = [self],
                         op = "<log>")
        # z = ln(x) -> dz/dself = e^f(x)
        self.children.append((1/self.val, z))
        return z

    def logarithm(self, a):
        if a == 1 or a <= 0:
            raise ValueError("The base should be any positive value != 1")
        z = VariableNode(np.log(self.val) / np.log(a),
                         parents = [self],
                         op = "<log_" + str(a) + ">")
        self.children.append((1 / (self.val * np.log(a)), z))
        return z

    def logistic(self):
        g_x = 1 / (1 + np.exp(-self.val))
        z = VariableNode(g_x,
                         parents = [self],
                         op = "<logistic>")
        self.children.append(((1 - g_x) * g_x, z))
        return z

    def sin(self):
        z = VariableNode(np.sin(self.val),
                         parents = [self],
                         op = "<sin>")
        # dz/dself = sin'(x) = cos(x)
        self.children.append((np.cos(self.val), z))
        return z

    def cos(self):
        z = VariableNode(np.cos(self.val),
                         parents = [self],
                         op = "<cos>")
        # dz/dself = cos'(x) = -sin(x)
        self.children.append((-np.sin(self.val), z))
        return z

    def tan(self):
        if self.val % np.pi == np.pi / 2:
            raise ValueError("input of tan should not be value of k * pi + pi/2")
        z = VariableNode(np.tan(self.val),
                         parents = [self],
                         op = "<tan>")
        # dz/dself = tan'(x) = 1/(cos(x))^2
        self.children.append((1 / (np.cos(self.val)) ** 2, z))
        return z

    def sinh(self):
        z = VariableNode((np.exp(self.val) - np.exp(-self.val)) / 2,
                         parents = [self],
                         op = "<sinh>")
        # dz/dself = sinh'(x) = cosh(x)
        self.children.append((np.cosh(self.val), z))
        return z

    def cosh(self):
        z = VariableNode((np.exp(self.val) + np.exp(-self.val)) / 2,
                         parents = [self],
                         op = "<cosh>")
        # dz/dself = cosh'(x) = sinh(x)
        self.children.append((np.sinh(self.val), z))
        return z

    def tanh(self):
        z = VariableNode(np.sinh(self.val) / np.cosh(self.val),
                         parents = [self],
                         op = "<tanh>")
        # dz/dself = tanh'(x) = 1 / cosh(x)^2
        self.children.append((1 / (np.cosh(self.val) ** 2), z))
        return z

    def arcsin(self):
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arcsin should within (-1, 1)")
        z = VariableNode(np.arcsin(self.val),
                         parents = [self],
                         op = "<arcsin>")
        # dz/dself = arcsin'(x) = 1/sqrt(1-x^2) * x'
        self.children.append((1/np.sqrt(1 - self.val**2), z))
        return z

    def arccos(self):
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arccos should within (-1, 1)")
        z = VariableNode(np.arccos(self.val),
                         parents = [self],
                         op = "<arccos>")
        # dz/dself = arccos'(x) = -1/sqrt(1-x^2) * x'
        self.children.append((-1/np.sqrt(1 - self.val**2), z))
        return z

    def arctan(self):
        z = VariableNode(np.arctan(self.val),
                         parents = [self],
                         op = "<arctan>")
        # dz/dself = arctan'(x) = 1/(1+x^2)
        self.children.append((1/(1 + self.val**2), z))
        return z

class Constant(VariableNode):

    def __init__(self, val):
        super(Constant, self).__init__(val, op='<Const>')

    def compute_grad(self):
        self._grad = 1