# function that evaluates polynomials given as lists of coefficients

x = 2
y = 0

coeffs = [3]
powers = [4]

for i in range(0, len(coeffs)): 
    y += coeffs[i] * x ** powers[i]

y

x = 2
y = 0

coeffs = [3,2,7,9,7]
powers = [4,3,2,1,0]

for i in range(0, len(coeffs)): 
    y += coeffs[i] * x ** powers[i]

import numpy as np
# lets evaluate at multiple points in one go
x = np.array([0, 1, 2, 3])
y = np.zeros(len(x))

coeffs = np.array([3,2,7,9,7])
powers = np.array([4,3,2,1,0])

for i in range(0, len(coeffs)):
    y += coeffs[i] * x ** powers[i]
