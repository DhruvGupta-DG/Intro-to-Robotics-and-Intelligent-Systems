
import control as co
import matplotlib.pyplot as plt
g= co.TransferFunction(1,(2,1,1))
t,y=co.step_response(g)
print(g)
plt.plot(t,y)
ax= plt.subplot()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


