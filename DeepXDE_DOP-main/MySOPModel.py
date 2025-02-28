import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

torch.set_default_device("cuda")

#I, e = 5.0607452392578125, 5.38771057128906
#r_1, m_1, h_1 = 0, 0.3257973861694336, 0.9999998807907104
#r_2, h_2, m_2, c1 = 14.617637634277344, 0.00015718663053121418, 0.11928328831345425, 0.00583516142796725
#r_3, h_3, m_3, c2 = 15.480697631835938, 0.002576290862634778, 0.1932005579874385, 0.0010395777644589543

I  = 3.0246481895446777
e  = 29.02496910095215
r_1= 12.978781700134277
m_1= 0.1679152369499207
h_1= 4.3882420868612826e-05
r_2= 0.00012770290777552873
h_2= 0.34682953357696533
m_2= 0.48459869623184204
c1 = 6.452714842453133e-06
r_3= 3.446214759605937e-05
h_3= 0.09573297202587128
m_3= 0.7626517415046692
c2 = 3.0134206099319272e-05

n00 = 140 #0.9 Steady state
n10 = 40 #0.5 Steady state
n20 = 1
n30 = 1

t_initial = 0
t_final = 10

def ode(t, Y):
    n0 = Y[:, 0:1]
    n1 = Y[:, 1:2]
    n2 = Y[:, 2:3]
    n3 = Y[:, 3:4]

    dn0_dt = dde.grad.jacobian(Y, t, i=0)
    dn1_dt = dde.grad.jacobian(Y, t, i=1)
    dn2_dt = dde.grad.jacobian(Y, t, i=2)
    dn3_dt = dde.grad.jacobian(Y, t, i=3)

    first  = dn0_dt 
    second = dn1_dt - n1 * n0 * r_1 + n1 * m_1 + (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) + (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1)
    third  = dn2_dt - (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) - n2 * m_2 - n2 * n3 * c1
    fourth = dn3_dt - (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1) - n3 * m_3 - n2 * n3 * c2

    #first = dn0_dt - n0
    #second = dn1_dt + n1
    #third = dn2_dt - n2
    #fourth = dn3_dt + n3

    return [first, second, third, fourth]

geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(x, on_initial):
    return on_initial

ic_n0 = dde.icbc.IC(geom, lambda x: n00, boundary, component=0)
ic_n1 = dde.icbc.IC(geom, lambda x: n10, boundary, component=1)
ic_n2 = dde.icbc.IC(geom, lambda x: n20, boundary, component=2)
ic_n3 = dde.icbc.IC(geom, lambda x: n30, boundary, component=3)

data = dde.data.TimePDE(geom, ode, [ic_n0, ic_n1, ic_n2, ic_n3], num_domain=15000, num_boundary=5000)

neurons = 40
layers = 12
layer_size = [1] + [neurons] * layers + [4]

activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)

losshistory, train_state = model.train(iterations=2000, display_every=100)

dde.utils.external.plot_loss_history(losshistory)
plt.show()

t_array = np.linspace(t_initial, t_final, 2048)

pinn_pred = model.predict(t_array.reshape(-1, 1))

n0_pinn = pinn_pred[:, 0:1]
n1_pinn = pinn_pred[:, 1:2]
n2_pinn = pinn_pred[:, 2:3]
n3_pinn = pinn_pred[:, 3:4]

t_test = np.array([[t_initial]])  # Check only at initial time
predicted_initial = model.predict(t_test)
print("Predicted Initial Conditions:", predicted_initial)


plt.plot(t_array, n0_pinn.flatten(), color="green", label=r"$x(t)$ PINNs, basal")
plt.plot(t_array, n1_pinn.flatten(), color="blue", label=r"$y(t)$ PINNs, prey")
plt.plot(t_array, n2_pinn.flatten(), color="red", label=r"$y(t)$ PINNs, predator 1")
plt.plot(t_array, n3_pinn.flatten(), color="orange", label=r"$y(t)$ PINNs, predator 2")


plt.legend()
plt.ylabel(r"population")
plt.xlabel(r"$t$")
plt.title("4 Species Predator-Prey Model")
plt.show()