import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

if torch.cuda.is_available():
    torch.set_default_device("cuda")


I   = 1000
r_1 = 2.1
m_1 = 0.0001 
h_1 = 0.9
r_2 = 0.9 
h_2 = 0.1 
r_3 = 1.2
h_3 = 0.01
m_2 = 0.008
m_3 = 0.004
c1  = 0.002
c2  = 0.004

n00 = 1000 #0.9 Steady state
n10 = 500 #0.5 Steady state
n20 = 15
n30 = 10

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

    first  = dn0_dt - I - math.e * n0 + (n1*n0*r_1)
    second = dn1_dt - n1 * n0 * r_1 + n1 * m_1 + (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) + (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1)
    third  = dn2_dt - (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) - n2 * m_2 - n2 * n3 * c1
    fourth = dn3_dt - (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1) - n3 * m_3 - n2 * n3 * c2

    return [first, second, third, fourth]

geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(_, on_initial):
    return on_initial

ic_n0 = dde.icbc.IC(geom, lambda x: n00, boundary, component=0)
ic_n1 = dde.icbc.IC(geom, lambda x: n10, boundary, component=1)
ic_n2 = dde.icbc.IC(geom, lambda x: n20, boundary, component=2)
ic_n3 = dde.icbc.IC(geom, lambda x: n30, boundary, component=3)


data = dde.data.PDE(geom, ode, [ic_n0, ic_n1, ic_n2, ic_n3], num_domain=512, num_boundary=4)

neurons = 64
layers = 32
layer_size = [1] + [neurons] * layers + [4]

activation = "ReLU"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)

losshistory, train_state = model.train(iterations=50000, display_every=500)

dde.utils.external.plot_loss_history(losshistory)
plt.show()

t_array = np.linspace(t_initial, t_final, 100)

pinn_pred = model.predict(t_array.reshape(-1, 1))

n0_pinn = pinn_pred[:, 0:1]
n1_pinn = pinn_pred[:, 1:2]
n2_pinn = pinn_pred[:, 2:3]
n3_pinn = pinn_pred[:, 3:4]


plt.plot(t_array, n0_pinn, color="green", label=r"$x(t)$ PINNs, basal")
plt.plot(t_array, n1_pinn, color="blue", label=r"$y(t)$ PINNs, prey")
plt.plot(t_array, n2_pinn, color="red", label=r"$y(t)$ PINNs, predator 1")
plt.plot(t_array, n3_pinn, color="orange", label=r"$y(t)$ PINNs, predator 2")


plt.legend()
plt.ylabel(r"population")
plt.xlabel(r"$t$")
plt.title("Lotka-Volterra numerical solution using PINNs method")
plt.show()