import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

if torch.cuda.is_available():
    torch.set_default_device("cuda")

def ode(x, y):
    #math.pow(t,2) - 2*math.pow(math.e, -2*t))
    dy_x = dde.grad.jacobian(y, x)
    return dy_x 

geom = dde.geometry.Interval(0, 10)

def boundary(x, on_boundary):
    return on_boundary

def boundary_func(x):
    return 1

def exact_sol(x):
    return np.sin(x) + 1

bc = dde.icbc.DirichletBC(geom, boundary_func, boundary)
data = dde.data.PDE(geom, ode, bc, num_domain=10, num_boundary=2, solution=exact_sol, num_test=1000)

neurons = 64
layers = 6
layer_size = [1] + [neurons] * layers + [1]

activation = "tanh"
initilizer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initilizer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000, display_every=1000)

dde.utils.external.plot_loss_history(losshistory)
plt.show()

x = geom.uniform_points(100, True)
y_pred = model.predict(x)
y_exact = exact_sol(x)
plt.figure()
plt.plot(x, y_pred, color='b', label='Predicted PINN Solution')
plt.plot(x, y_exact, '*', color='r', label='Exact Solution')
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.legend(loc='best')
plt.show()