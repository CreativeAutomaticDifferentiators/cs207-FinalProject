# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:56:02 2019

@author: pelan
"""

import forward

x = Var(0.12, ['x'])
y = Var(1.2, ['y'])

z = ln(cos(x)*sin(y+2*x))
print('value of the expression', z.value)
print('gradient', z.grad)
