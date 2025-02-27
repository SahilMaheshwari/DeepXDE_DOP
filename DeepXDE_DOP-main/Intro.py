import numpy
print('there')

# Set the backend before importing DeepXDE
import os
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
print('somewhere')

from deepxde.backend import tf
print('here')


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
activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN([1] + [neurons] * layers + [2], activation, initialiser)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=50000, display_every=10000)

dde.utils.external.plot_loss_history(losshistory)

pinn_pred = model.predict(t_array.reshape(-1, 1))
x_pinn = pinn_pred[:, 0:1]
y_pinn = pinn_pred[:, 1:2]