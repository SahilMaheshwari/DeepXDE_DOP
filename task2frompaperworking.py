import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

if torch.cuda.is_available():
    torch.set_default_device("cuda")



alpha = 2/3 #reprodcution rate of x
beta = 4/3 #loss of x cause of y
gamma = 1 #gain of y cause of x
delta = 1 #death rate of y

x0 = 1.2 #0.9 Steady state
y0 = 0.8 #0.5 Steady state

t_initial = 0
t_final = 10

def ode(t, Y):
    x = Y[:, 0:1]
    y = Y[:, 1:2]
    dx_dt = dde.grad.jacobian(Y, t, i=0)
    dy_dt = dde.grad.jacobian(Y, t, i=1)
    return [dx_dt - alpha * x + beta * x * y, dy_dt + gamma * y - delta * x * y]

geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(_, on_initial):
    return on_initial

ic_x = dde.icbc.IC(geom, lambda x: x0, boundary, component=0)
ic_y = dde.icbc.IC(geom, lambda x: y0, boundary, component=1)

data = dde.data.PDE(geom, ode, [ic_x, ic_y], num_domain=512, num_boundary=2)

neurons = 64
layers = 6
layer_size = [1] + [neurons] * layers + [2]

activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)

losshistory, train_state = model.train(iterations=1000, display_every=100)

#dde.utils.external.plot_loss_history(losshistory)
#plt.show()

t_array = np.linspace(t_initial, t_final, 100)
pinn_pred = model.predict(t_array.reshape(-1, 1))
x_pinn = pinn_pred[:, 0:1]
y_pinn = pinn_pred[:, 1:2]
plt.plot(t_array, x_pinn, color="green", label=r"$x(t)$ PINNs, prey")
plt.plot(t_array, y_pinn, color="blue", label=r"$y(t)$ PINNs, predator")
plt.legend()
plt.ylabel(r"population")
plt.xlabel(r"$t$")
plt.title("Lotka-Volterra numerical solution using PINNs method")
plt.show()