
import control as co
import matplotlib.pyplot as plt

g= co.TransferFunction(1,(1,1,1))
t,y=co.step_response(g)
print(g)
plt.plot(t,y)
(num,den) = co.pade(0.25,3)
Gp = co.tf(num,den)*g
print(Gp)
mag,phase,omega = co.bode(Gp)
plt.show()









