import numpy as np

class AutoDiffFwd:
    def __init__(self):
        self.var2idx = {}

    def createVariables(self, names, vals):
        """Create all variables needed in the following computation

            INPUTS
            =======
            names: variables name ['x', 'y']
            vals: variables value [1, 2]

            RETURNS
            ========
            record the relationship between index and var name
            Returns a list of Variable objects

            EXAMPLES
            =========
            >>> autodiff = AutoDiffFwd()
            >>> x, y = autodiff.createVariables(['x', 'y'], [1, 2])
            >>> print(x)
            >>> print(y)
            val: 1 der: [ 1.  0.]
            val: 2 der: [ 0.  1.]
        """
        list_of_variables = []
        num_variables = len(vals)
        for i in range(num_variables):
            ders = np.zeros(num_variables)
            ders[i] = 1
            self.var2idx[names[i]] = i
            list_of_variables.append(Variable(vals[i], ders))
        return list_of_variables

    def reset(self):
        """Clear the variables recorded in order to be reused in new computation
        """
        self.var2idx = {}
        return

    def createVectorFunction(self, list_of_variables):
        """Combine multiple functions into an vector function

            INPUTS
            =======
            list_of_variables: list of functions

            RETURNS
            ========
            Returns a new VectorVariable object

            EXAMPLES
            =========
            >>> autodiff = AutoDiffFwd()
            >>> x, y = autodiff.createVariables(['x', 'y'], [1, 2])
            >>> f1 = x * (y ** 2)
            >>> f2 = np.sin(x * y)
            >>> vec_f = autodiff.createVectorFunction([f1, f2])
            >>> print(vec_f)
            val: [ 4,  0.909] der: [[ 4    , 4     ]
                                    [-0.832, -0.416]]
        """
        vals = np.array([var.val for var in list_of_variables])
        jacobian = np.stack([var.der for var in list_of_variables], axis=0)
        return VectorVariable(vals, jacobian)

    def getVerboseInformation(self, var):
        """Print verbose information related to given variable
        """
        print(var)
        if isinstance(var.der, (int, float)) or var.der.shape[0] == 1:
            print("Current function is an univariate function")
        elif var.der.shape[0] > 1:
            print("Current function is an multivariate function")
        if len(var.der.shape) > 1:
            print("Current function is an vector function")
        for var_name, i in self.var2idx.items():
            print("partial derivative with respect of " + var_name + " is " + str(var.der[:,i]))


class Variable:
    def __init__(self, val, der=1):
        """
        val: int/float, value of variable for differentiation
        der: int/float or np.array if mutiple variable
             (it is just a scalar for this milestone),
             partial derivatives of variables for differentiation
             each entry represents an partial derivatives (∂u/∂x, ∂u/∂y, ∂u/∂z, ...)
        """
        self.val = val
        self.der = der

    # Comparison Operators

    def __eq__(self, other):
        """Check if two variables are equal

            INPUTS
            =======
            self: Variable
            other: Variable

            RETURNS
            ========
            Returns True if both val and der of two variables are equal.
            Otherwise False

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = Variable(2)
            >>> print(x == y)
            True
        """
        try:
            # two variable would be equal if and only if their val and der are equal
            return (self.val == other.val and np.array_equal(self.der, other.der))
        except:
            # also return False if other is not a valid Variable
            return False

    def __ne__(self, other):
        """Check if two variables are equal

            INPUTS
            =======
            self: Variable
            other: Variable

            RETURNS
            ========
            Returns False if both val and der of two variables are equal.
            Otherwise True

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = Variable(2)
            >>> print(x != y)
            False
        """
        return not (self == other)

    # Basic Operations

    def __add__(self, other):
        """Returns self + other.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents self + other
            new val = self.val + other.val
            new der = self.der + other.der
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x + 1
            Variable(3, 1)
        """
        if isinstance(other, (int, float)):
            # other is a number
            return Variable(self.val + other, self.der)
        # assume other is a valid Variable, otherwise raise Error
        try:
            return Variable(self.val + other.val, self.der + other.der)
        except:
            # other error: derivatives dimensions does not match
            raise TypeError("other is not a valid Variable or number.")

    def __radd__(self, other):
        """Returns other + self.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents other + self
            new val = self.val + other.val
            new der = self.der + other.der
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = 1 + x
            Variable(3, 1)
        """
        return self + other

    def __neg__(self):
        """Returns -self.

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents -self.
            new val = -self.val
            new der = -self.der

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = -x
            Variable(-2, -1)
        """
        return Variable(-self.val, -self.der)

    def __sub__(self, other):
        """Returns self - other.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents self - other
            new val = self.val - other.val
            new der = self.der - other.der
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x - 1
            Variable(1, 1)
        """
        return self + (-other)

    def __rsub__(self, other):
        """Returns other - self.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents other - self
            new val = other.val - self.val
            new der = other.der - self.der
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = 1 - x
            Variable(-1, -1)
        """
        return other + (-self)

    def __mul__(self, other):
        """Returns self * other.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents self * other
            new val = self.val * other.val
            new der = self.der * other (if other is number)
            new der = f'(x)g(x) + f(x)g'(x) (if other is Variable)
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x * 2
            Variable(4, 2)
        """
        try:
            # other is a number
            if isinstance(other, (int, float)):
                return Variable(self.val * other, self.der * other)
            # assume other is a valid Variable, otherwise raise Error
            else:
                # product rule for derivative of functions
                # d/dx(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
                return Variable(self.val * other.val, self.der * other.val + self.val * other.der)
        except:
            # other error: derivatives dimensions does not match
            raise TypeError("other is not a valid Variable or number.")

    def __rmul__(self, other):
        """Returns other * self.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents self * other
            new val = self.val * other.val
            new der = self.der * other (if other is number)
            new der = f'(x)g(x) + f(x)g'(x) (if other is Variable)
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x * 2
            Variable(4, 2)
        """
        return self * other

    def __truediv__(self, other):
        """Returns self / other.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents self / other
            new val = self.val / other.val
            new der = self.der / other (if other is number)
            new der = f'(x)g(x) - f(x)g'(x)) / (g(x)^2 (if other is Variable)
            (Error exception is raised when the input is not valid)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x / 2
            Variable(1, 1/2)
        """
        try:
            # other is a number
            if isinstance(other, (int, float)):
                return Variable(self.val / other, self.der / other)
            # assume other is a valid Variable, otherwise raise Error
            else:
                # Quotient rule for derivative of functions
                # d/dx(f(x)/g(x)) = f'(x)g(x) - f(x)g'(x)) / (g(x)^2
                return Variable(self.val / other.val,
                                (self.der * other.val - self.val * other.der) / (other.val)**2)
        except ZeroDivisionError:
            raise ZeroDivisionError("Failed when dividing by 0")
        except:
            # other error: derivatives dimensions does not match
            raise TypeError("other is not a valid Variable or number.")

    def __rtruediv__(self, other):
        """Returns other / self.

            INPUTS
            =======
            self: Variable object
            other: Variable or float/int

            RETURNS
            ========
            a new Variable represents other / self
            new val = other.val / self.val
            new der = -f'/(f^2) (if other is number)
            new der = f'(x)g(x) - f(x)g'(x)) / (g(x)^2 (if other is Variable)
            (Error exception is raised when the input is not valid or the Denominator is 0)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = 2/x
            Variable(1, -1/2)
        """
        try:
            # other is a number
            if isinstance(other, (int, float)):
                # Reciprocal Rule: 1/f = -f'/(f^2)
                return Variable(other / self.val,
                                (- other * self.der) / (self.val) ** 2)
                # assume other is a valid Variable, otherwise raise Error
            else:
                # Quotient rule for derivative of functions
                # d/dx(f(x)/g(x)) = f'(x)g(x) - f(x)g'(x)) / (g(x)^2
                return Variable(other.val / self.val,
                                (other.der * self.val - other.val * self.der) / (self.val)**2)
        except ZeroDivisionError:
            raise ZeroDivisionError("Failed when dividing by 0")
        except:
            # other error: derivatives dimensions does not match
            raise TypeError("other is not a valid Variable or number.")

    def __pow__(self, power):
        """Returns self^power (x^p).

            INPUTS
            =======
            self: Variable object
            power: float/int

            RETURNS
            ========
            a new Variable represents x^p
            new val = self.val^power
            new der = d/dx(f(x)^n) = n(f(x)^(n-1)) * f'(x)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x**2
            Variable(4, 4)
        """
        return Variable(self.val ** power, power * self.val ** (power - 1) * self.der)

    def sqrt(self):
        """Returns sqrt(self) (x^0.5).

            INPUTS
            =======
            self: Variable object
            power: float/int

            RETURNS
            ========
            a new Variable represents x^0.5
            new val = self.val^0.5
            new der = d/dx(f(x)^n) = n(f(x)^(n-1)) * f'(x)

            EXAMPLES
            =========
            >>> x = Variable(4)
            >>> y = np.sqrt(x)
            Variable(2, 1/4)
        """
        return Variable(self.val, self.der) ** 0.5

    # Exponentials

    def __rpow__(self, base):
        """Returns base^(self) (a^x).

            INPUTS
            =======
            self: Variable object
            base: float/int

            RETURNS
            ========
            a new Variable represents a^x
            new val = base^self.val
            new der =  d/dx(a^f(x)) = ln(a) * a^f(x) * f'(x)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = 2**x
            Variable(4, 2.7725)
        """
        return Variable(base**self.val, np.log(base) * (base ** self.val) * self.der)

    def exp(self):
        """Returns e^(self) (e^x).

            INPUTS
            =======
            self: Variable object
            base: float/int

            RETURNS
            ========
            a new Variable represents e^x
            new val = e^self.val
            new der =  d/dx(a^f(x)) = e^f(x) * f'(x)

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.exp(x)
            Variable(7.389, 7.389)
        """
        return Variable(np.exp(self.val), np.exp(self.val) * self.der)

    # Logarithms

    def log(self):
        """Returns ln(self) (ln(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents ln(x)
            new val = ln(self.val)
            new der =  ln'(x) = 1/x * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.log(x)
            Variable(0.693, 0.5)
        """
        return Variable(np.log(self.val), 1 / (self.val) * (self.der))

    def logarithm(self, a):
        """Returns log_a(self) (log_a(x)).

            INPUTS
            =======
            self: Variable object
            a : log base

            RETURNS
            ========
            a new Variable represents log_a(x)
            new val = log_a(self.val)
            new der =  log_a'(x) = 1/(x * log_a(x)) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x.logarithm(2)
            Variable(1, 0.72134)
        """
        if a == 1 or a <= 0:
            raise ValueError("The base should be any positive value != 1")
        else:
            return Variable(np.log(self.val) / np.log(a), 1 / (self.val * np.log(a)) * self.der)

    # Logistic function

    def logistic(self):
        """Returns logistic(self).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents g(x)
            new val = g(x) = 1 / (1 + np.exp(-self.val))
            new der =  g(x) (1 - g(x)) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = x.logistic()
            Variable(0.88, 0.104)
        """
        g_x = 1 / (1 + np.exp(-self.val))
        return Variable(g_x, (1 - g_x) * g_x * self.der)

    # Trig functions

    def sin(self):
        """Returns sin(self) (sin(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents sin(x)
            new val = sin(self.val)
            new der = sin'(x) = cos(x) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.sin(x)
            Variable(0.909, -0.416)
        """
        return Variable(np.sin(self.val), np.cos(self.val) * self.der)

    def cos(self):
        """Returns cos(self) (cos(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents cos(x)
            new val = sin(self.val)
            new der = cos'(x) = -sin(x) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.cos(x)
            Variable(-0.416, -0.909)
        """
        return Variable(np.cos(self.val), -np.sin(self.val) * self.der)

    def tan(self):
        """Returns tan(self) (tan(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents cos(x)
            new val = tan(self.val)
            new der = tan'(x) = 1/(cos(x))^2 * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.tan(x)
            Variable(-2.185, 5.774)
        """
        if self.val % np.pi == np.pi / 2:
            raise ValueError("input of tan should not be value of k * pi + pi/2")
        return Variable(np.tan(self.val), 1 / (np.cos(self.val)) ** 2 * self.der)

    # Inverse trig functions

    def arcsin(self):
        """Returns arcsin(self) (arcsin(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents arcsin(x)
            new val = arcsin(self.val)
            new der = arcsin'(x) = 1/sqrt(1-x^2) * x'
            (Error would be raised if val is not within (-1,1))

            EXAMPLES
            =========
            >>> x = Variable(0.5)
            >>> y = np.arcsin(x)
            Variable(0.523, 1.154)
        """
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arcsin should within (-1, 1)")
        return Variable(np.arcsin(self.val), 1/np.sqrt(1 - self.val**2) * self.der)

    def arccos(self):
        """Returns arccos(self) (arccos(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents arccos(x)
            new val = arccos(self.val)
            new der = arccos'(x) = -1/sqrt(1-x^2) * x'
            (Error would be raised if val is not within (-1,1))

            EXAMPLES
            =========
            >>> x = Variable(0.5)
            >>> y = np.arccos(x)
            Variable(1.047, -1.154)
        """
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arccos should within (-1, 1)")
        return Variable(np.arccos(self.val), -1/np.sqrt(1 - self.val**2) * self.der)

    def arctan(self):
        """Returns arctan(self) (arctan(x)).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents arctan(x)
            new val = arccos(self.val)
            new der = arctan'(x) = 1/(1+x^2) * x'

            EXAMPLES
            =========
            >>> x = Variable(0.5)
            >>> y = np.arctan(x)
            Variable(0.463, 0.8)
        """
        return Variable(np.arctan(self.val), 1/(1 + self.val**2) * self.der)

    # Hyperbolic functions

    def sinh(self):
        """Returns sinh(self).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents sinh(x)
            new val = (e^(x) - e^(-x)) / 2
            new der = cosh(x) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.sinh(x)
            Variable(3.6268, 3.762)
        """
        return Variable((np.exp(self.val) - np.exp(-self.val)) / 2, np.cosh(self.val) * self.der)

    def cosh(self):
        """Returns cosh(self).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents cosh(x)
            new val = (e^(x) + e^(-x)) / 2
            new der = sinh(x) * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.cosh(x)
            Variable(3.762, 3.626)
        """
        return Variable((np.exp(self.val) + np.exp(-self.val)) / 2, np.sinh(self.val) * self.der)

    def tanh(self):
        """Returns tanh(self).

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            a new Variable represents tanh(x)
            new val = sinh(x) / cosh(x)
            new der = 1 / cosh(x)^2 * x'

            EXAMPLES
            =========
            >>> x = Variable(2)
            >>> y = np.tanh(x)
            Variable(0.964, 0.0706)
        """
        return Variable(np.sinh(self.val) / np.cosh(self.val), 1 / (np.cosh(self.val) ** 2) * self.der)

    def __str__(self):
        """Returns a string shows the val and der of the current Variable.

            INPUTS
            =======
            self: Variable object

            RETURNS
            ========
            "val: x der: x'"

            EXAMPLES
            =========
            >>> x = Variable(0.5)
            >>> print(x)
            val: 0.5 der: 1
        """
        return "val: " + str(self.val) + " der: " + str(self.der)

    def get_jacobian(self):
        """
            Get the Jacobian matrix of partial derivatives.
        """
        return self.der

class VectorVariable(Variable):
    def __init__(self, vals, jacobian):
        """
        val: np.array, values of variable for differentiation
        der: np.array (Jacobian matrix)
             matrix of partial derivatives of variables for differentiation
             each row represents an partial derivatives (∂f1/∂x, ∂f1/∂y, ∂f1/∂z, ...)
        """
        self.val = vals
        self.der = jacobian

    def get_jacobian(self):
        """
            Get the Jacobian matrix of partial derivatives.
        """
        return self.der

    # VectorVariable would only support scalar operation inherited from Variable

    def __eq__(self, other):
        """Check if two variables are equal
        """
        try:
            # two variable would be equal if and only if their val and der are equal
            return (np.array_equal(self.val, other.val) and np.array_equal(self.der, other.der))
        except:
            # also return False if other is not a valid Variable
            return False

    def __ne__(self, other):
        """Check if two variables are equal
        """
        return not (self == other)

    def __str__(self):
        """Returns a string shows the vals and jacobian matrix of the current vector Variable.
        """
        return "vals: " + str(self.val) + " jacobian: " + str(self.der)

