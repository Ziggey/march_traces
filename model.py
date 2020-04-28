import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for unequal plot boxes
from scipy.optimize import curve_fit
import pandas as pd

#def func(x, a, a1, a2, b, b1, b2, c1, c2):
#    return a * np.exp(-b * x) + a1 * np.exp(-(x - b1/c1)**2) + a2 * np.exp(-(x - b2/c2)**2)

df = pd.read_csv('h.265+mp3.csv')
x = df.Time.values
y = df.Jitter.values

'''
a, a1, a2, b, b1, b2, c1, c2 = 100, 200, 300, 14, 18, 20, 23, 31

# fit data using SciPy's Levenberg-Marquart method
nlfit, nlpcov = scipy.optimize.curve_fit(func,
                x, y, p0=[a, a1, a2, b, b1, b2, c1, c2])

a_, a1_, a2_, b_, b1_, b2_, c1_, c2_ = nlfit

plt.plot(x, func(x, a_, a1_, a2_, b_, b1_, b2_, c1_, c2_), 'r-', label='fit')
plt.plot(x, y, 'bo')
plt.show()
'''
n = len(x)
S = np.empty(n)

def yp(x, tau, theta):
    z = 5.0 * (1 - np.exp(-(x - theta) / tau))
    #theta = 0.4
    for i in range(n):
        if x[i]<theta:
            S[i] = 0
        else:
            S[i] = 1
    return z * S

#print (yp(t, 3,0.2))


c, cov = curve_fit(yp,x,y)

yopt = yp(x, c[0], c[1])
print(yopt)


from sklearn.metrics import r2_score
print('R^2: ', r2_score(yopt,y))

plt.plot(x, y, 'ro')
plt.plot(x,yp(x, 3,0.2),'b-', label='no curve fit')
plt.plot(x,yopt,'m-', label='curve fit')
plt.legend()
plt.show()