def DE(y, t):
    return (1 - t + 4*y)

def ForEuler(dY_dT, Y0, T0, T,iterations=100):

    step = (T - T0)/iterations
    Y_t = Y0
    t = T0

    for i in range(iterations):
        Y_tp1 = Y_t + step * dY_dT(Y_t, t)
        t += step
        Y_t = Y_tp1

    return Y_t

ans = ForEuler(DE, 1, 0, 0.1, 1)
print(ans)