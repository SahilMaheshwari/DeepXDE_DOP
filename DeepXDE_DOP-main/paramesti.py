import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

torch.set_default_device("cuda")

I =   torch.nn.Parameter(torch.tensor(4    , dtype=torch.float32))
e =   torch.nn.Parameter(torch.tensor(30   , dtype=torch.float32))

r_1 = torch.nn.Parameter(torch.tensor(12    , dtype=torch.float32))
m_1 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
h_1 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

r_2 = torch.nn.Parameter(torch.tensor(0.6 , dtype=torch.float32))
h_2 = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
m_2 = torch.nn.Parameter(torch.tensor(0.01 , dtype=torch.float32))
c1  = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

r_3 = torch.nn.Parameter(torch.tensor(0.05  , dtype=torch.float32))
h_3 = torch.nn.Parameter(torch.tensor(0.04, dtype=torch.float32))
m_3 = torch.nn.Parameter(torch.tensor(0.07 , dtype=torch.float32))
c2  = torch.nn.Parameter(torch.tensor(0.02 , dtype=torch.float32))

n00 = 2500 #0.9 Steady state
n10 = 1600 #0.5 Steady state
n20 = 60
n30 = 40

t_initial = 0
t_final = 5

def ode(t, Y):
    n0 = Y[:, 0:1]
    n1 = Y[:, 1:2]
    n2 = Y[:, 2:3]
    n3 = Y[:, 3:4]

    dn0_dt = dde.grad.jacobian(Y, t, i=0)
    dn1_dt = dde.grad.jacobian(Y, t, i=1)
    dn2_dt = dde.grad.jacobian(Y, t, i=2)
    dn3_dt = dde.grad.jacobian(Y, t, i=3)

    #first  = dn0_dt - I - math.e * n0 + (n1*n0*r_1)
    #second = dn1_dt - n1 * n0 * r_1 + n1 * m_1 + (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) + (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1)
    #third  = dn2_dt - (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) - n2 * m_2 - n2 * n3 * c1
    #fourth = dn3_dt - (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1) - n3 * m_3 - n2 * n3 * c2

    first  = dn0_dt - torch.abs(I) - torch.abs(e) * n0 + (n1*n0*torch.abs(r_1))/(1 + torch.abs(r_1)* torch.abs(h_1) * n0)
    second = dn1_dt - n1 * n0 * torch.abs(r_1)/(1 + torch.abs(r_1)* torch.abs(h_1) * n0) + n1 * torch.abs(m_1) + (n2 * n1 * torch.abs(r_2))/(1 + torch.abs(h_2) * torch.abs(r_2) * n1) + (n3 * n1 * torch.abs(r_3))/(1 + torch.abs(h_3) * torch.abs(r_3) * n1)
    third  = dn2_dt - (n2 * n1 * torch.abs(r_2))/(1 + torch.abs(h_2) * torch.abs(r_2) * n1) - n2 * torch.abs(m_2) - n2 * n3 * torch.abs(c1)
    fourth = dn3_dt - (n3 * n1 * torch.abs(r_3))/(1 + torch.abs(h_3) * torch.abs(r_3) * n1) - n3 * torch.abs(m_3) - n2 * n3 * torch.abs(c2)

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

data = dde.data.TimePDE(geom, ode, [ic_n0, ic_n1, ic_n2, ic_n3], num_domain=5000, num_boundary=5000)

neurons = 64
layers = 12
layer_size = [1] + [neurons] * layers + [4]

activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)

optimizer = torch.optim.Adam([I, e, r_1, m_1, h_1, r_2, h_2, m_2, c1, r_3, h_3, m_3, c2], lr=0.0005)

def custom_loss(y_true, y_pred):
    n0, n1, n2, n3 = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3], y_pred[:, 3:4]
    penalty_nonneg = (
        100*torch.square(torch.sum(torch.relu(-n0 + 0.01))) +
        20*torch.square(torch.sum(torch.relu(-n1 + 0.01))) +
        5*torch.square(torch.sum(torch.relu(-n2 + 0.01))) +
        5*torch.square(torch.sum(torch.relu(-n3 + 0.01)))
    )

    penalty_morepred = (
        torch.square(torch.sum(torch.relu(n1 - n2))) +
        torch.square(torch.sum(torch.relu(n1 - n3))) +
        torch.square(torch.sum(torch.relu(n0 - n1)))
    )

    reward_highpop = torch.min(
        ((10*torch.square(torch.sum(torch.relu(n0 - 2.5))) +
        2*torch.square(torch.sum(torch.relu(n1 - 1.5))) +
        torch.square(torch.sum(torch.relu(n2 - 0.5))) +
        torch.square(torch.sum(torch.relu(n3 - 0.5))))),
        torch.tensor(1e+03)
    )

    penalty_badparameters = (
        5*torch.pow(torch.max(torch.relu(-m_1), torch.relu(m_1 - 1)), 3) +
        5*torch.pow(torch.max(torch.relu(-m_2), torch.relu(m_2 - 1)), 3) +
        5*torch.pow(torch.max(torch.relu(-m_3), torch.relu(m_3 - 1)), 3) +
        5*torch.pow(torch.max(torch.relu(-c1), torch.relu(c1 - 1)), 3) +
        5*torch.pow(torch.max(torch.relu(-c2), torch.relu(c2 - 1)), 3)
    )

    penalty = (
        5000 * penalty_nonneg +
        10 * penalty_morepred +
        5000 * penalty_badparameters -
        100 * reward_highpop
    )

    loss = torch.mean(torch.square(y_true - y_pred)) + penalty
    loss = torch.relu(loss)

    return loss

model.compile(optimizer, loss=custom_loss)

checker = dde.callbacks.ModelCheckpoint(
    r"C:\Users\Sahil\Downloads\DeepXDE_DOP-main\DeepXDE_DOP-main\models\model200kREAAALLLLLBABYYYY.ckpt", save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=1000, display_every=1, callbacks=[checker], model_restore_path = r"C:\Users\Sahil\Downloads\DeepXDE_DOP-main\DeepXDE_DOP-main\models\model200kREAAALLLLLBABYYYY.ckpt-53000.pt")

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
plt.title("4 species prey-predator model")
plt.show()

print(f"Estimated I  : {I.item()}")
print(f"Estimated e  : {e.item()}")
print(f"Estimated r_1: {r_1.item()}")
print(f"Estimated m_1: {m_1.item()}")
print(f"Estimated h_1: {h_1.item()}")
print(f"Estimated r_2: {r_2.item()}")
print(f"Estimated h_2: {h_2.item()}")
print(f"Estimated m_2: {m_2.item()}")
print(f"Estimated c1 : {c1.item()}")
print(f"Estimated r_3: {r_3.item()}")
print(f"Estimated h_3: {h_3.item()}")
print(f"Estimated m_3: {m_3.item()}")
print(f"Estimated c2 : {c2.item()}")