import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#https://www.youtube.com/watch?v=uoshfzYW6vA&t=47s

x = pd.read_csv('https://apmonitor.com/che263/uploads/Main/dynamics.txt')

#print(x.head())

t = x['time (min)'].values
y = x['y'].values

n = len(t)
S = np.empty(n)
# or S = np.empty_like(t)




def yp(t, tau, theta):
    z = 5.0 * (1 - np.exp(-(t - theta) / tau))
    #theta = 0.4
    for i in range(n):
        if t[i]<theta:
            S[i] = 0
        else:
            S[i] = 1
    return z * S

#print (yp(t, 3,0.2))


c, cov = curve_fit(yp,t,y)

yopt = yp(t, c[0], c[1])
print(yopt)


from sklearn.metrics import r2_score
print('R^2: ', r2_score(yopt,y))

plt.plot(t, y, 'ro')
plt.plot(t,yp(t, 3,0.2),'b-', label='no curve fit')
plt.plot(t,yopt,'m-', label='curve fit')
plt.legend()
plt.show()