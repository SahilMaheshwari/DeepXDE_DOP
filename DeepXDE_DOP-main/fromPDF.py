def ode(t, Y):
    x = Y[:, 0:1]
    y = Y[:, 1:2]
    dx_dt = dde.grad.jacobian(Y, t, i=0)
    dy_dt = dde.grad.jacobian(Y, t, i=1)
    return [dx_dt - alpha * x + beta * x * y, dy_dt + gamma * y - delta * x * y]