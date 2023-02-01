
import control as co
import matplotlib.pyplot as plt

g= co.TransferFunction(1,(1,1,1))
t,y=co.step_response(g)
print(g)

plt.plot(t,y)
plt.show()









