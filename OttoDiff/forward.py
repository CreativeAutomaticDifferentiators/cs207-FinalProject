import numpy as np

class Variable:
    def __init__(self, val, der=1):
        '''
        val: int/float, np.array. if multiple values
            values of variables for differentiation
        der: it is just a scalar for this milestone,
            np.array for future, which is partial derivatives of variables for differentiation
            each entry represents an partial derivatives (∂u/∂x, ∂u/∂y, ∂u/∂z)
        '''
        self.val = val
        self.der = der

    # basic operations (addition, multiplication, subtraction, division, power)

    def __add__(self, other):
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
        return self + other

    def __neg__(self):
        return Variable(-self.val, -self.der)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        '''
            self * other
        '''
        try:
            # other is a number
            if isinstance(other, (int, float)):
                return Variable(self.val * other, self.der  * other)
            # assume other is a valid Variable, otherwise raise Error
            else:
                # product rule for derivative of functions
                # d/dx(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
                return Variable(self.val * other.val, self.der * other.val + self.val * other.der)
        except:
            # other error: derivatives dimensions does not match
            raise TypeError("other is not a valid Variable or number.")

    def __rmul__(self, other):
        '''
            other * self
        '''
        return self * other

    def __truediv__(self, other):
        '''
        self/other
        '''
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
        '''
            other/self
        '''
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
        '''
            x^p
            Chain Rule: d/dx(f(x)^n) = n(f(x)^(n-1)) * f'(x)
        '''
        return Variable(self.val ** power, power * self.val ** (power - 1) * self.der)

    def __rpow__(self, base):
        '''
            a^x
            Chain Rule: d/dx(a^f(x)) = ln(a) * a^f(x) * f'(x)
        '''
        return Variable(base**self.val, np.log(base) * (base ** self.val) * self.der)

    def exp(self):
        return (np.e)**self

    def log(self):
        '''
            np.log(x)
            Chain Rule: ln'(x) = 1/x * x'
        '''
        return Variable(np.log(self.val), 1 / (self.val) * (self.der))

    # -trig functions (sine, cosine, tangent)
    def sin(self):
        '''
            sin(x)
            Chain Rule: sin'(x) = cos(x) * x'
        '''
        return Variable(np.sin(self.val), np.cos(self.val) * self.der)

    def cos(self):
        '''
            sin(x)
            Chain Rule: cos'(x) = sin(x) * x'
        '''
        return Variable(np.cos(self.val), -np.sin(self.val) * self.der)

    def tan(self):
        '''
            tan(x)
            Chain Rule: tan'(x) = 1/(cos(x))^2 * x'
        '''
        if self.val % np.pi == np.pi / 2:
            raise ValueError("input of tan should not be value of k * pi + pi/2")
        return Variable(np.tan(self.val), 1 / (np.cos(self.val)) ** 2 * self.der)

    def arcsin(self):
        '''
            arcsin(x)
            Chain Rule: 1/sqrt(1-x^2) * x'
        '''
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arcsin should within (-1, 1)")
        return Variable(np.arcsin(self.val), 1/np.sqrt(1 - self.val**2) * self.der)

    def arccos(self):
        '''
            arccos(x)
            Chain Rule: -1/sqrt(1-x^2) * x'
        '''
        if self.val <= -1 or self.val >= 1:
            raise ValueError("input of arccos should within (-1, 1)")
        return Variable(np.arccos(self.val), -1/np.sqrt(1 - self.val**2) * self.der)

    def arctan(self):
        '''
            arctan(x)
            Chain Rule: 1/(1+x^2) * x'
        '''
        return Variable(np.arctan(self.val), 1/(1 + self.val**2) * self.der)

    def __str__(self):
        return "val: " + str(self.val) + " der: " + str(self.der)

    def get_jacobian(self):
        '''
            Get the Jacobian matrix of partial derivatives.
            For this milestone, the Jacobian will just be a scalar.
        '''
        return self.der