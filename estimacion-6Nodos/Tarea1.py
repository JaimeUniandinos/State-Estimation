import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,10,0.1)
y=2*np.sin(4*x)-x**2+10*x
plt.plot(x,y)
plt.show()