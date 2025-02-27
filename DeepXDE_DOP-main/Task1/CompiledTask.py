import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import inspect

def DE(t, y):
    #return (3 + math.pow(math.e, -t) - 0.5*y)
    #return (math.pow(t,2) - 2*math.pow(math.e, -2*t))
    #return (t + y*t)/10
    return (math.cos(t))

def soln(y0, t0, t):
    #return 6 - 2*math.pow(math.e, -t) - 3 * math.pow(math.e, -t/2)
    #return (math.pow(t,3)/3) + math.pow(math.e, -2*t)
    #return ((1+y0)* math.pow(math.e, (math.pow(t,2)-math.pow(t0, 2))/20)) - 1
    return (math.sin(t) + 1)

def ForEuler(dY_dT, Y0, T0, T,iterations=100):

    iterArr = []
    step = (T - T0)/iterations
    Y_t = Y0
    t = T0

    for i in range(iterations):

        iterArr.append(Y_t)
        Y_t = Y_t + step * dY_dT(t, Y_t)
        t += step
    iterArr.append(Y_t)

    return Y_t, iterArr

def RK2(dY_dT, Y0, T0, T,iterations=100):
    
    iterArr = []
    step = (T - T0)/iterations
    Y_t = Y0
    t = T0

    for i in range(iterations):
        iterArr.append(Y_t)
        k1 = step * dY_dT(t, Y_t)
        k2 = step * dY_dT(t + step ,Y_t + k1)

        Y_t = Y_t + 0.5*(k1 + k2)
        t += step
    iterArr.append(Y_t)
        
    return Y_t, iterArr

def RK4(dY_dT, Y0, T0, T,iterations=100):
    
    iterArr = []
    step = (T - T0)/iterations
    Y_t = Y0
    t = T0

    for i in range(iterations):
        iterArr.append(Y_t)
        k1 = step * dY_dT(t, Y_t)

        k2 = step * dY_dT(t + 0.5*step, Y_t + 0.5*k1)

        k3 = step * dY_dT(t + 0.5*step, Y_t + 0.5*k2)

        k4 = step * dY_dT(t + step, Y_t + k3)

        Y_t = Y_t + (k1 + 2*k2 + 2*k3 + k4)/6
        t += step
     
    iterArr.append(Y_t)
    return Y_t, iterArr

def errorFinder(soln, estimates):
    errors = []

    for i in estimates:
        errors.append((abs(soln-i))/soln)

    relerror = []
    for i in range(len(errors) - 1):
         relerror.append( errors[i+1] / errors[i] )

def main(dY_dT, Y0, T0, T, iterations, showgraph, showsoln=False):

    h = (T - T0)/ iterations
    t_array = np.linspace(T0, T, iterations + 1)
    try:
        solveIVP = solve_ivp(DE, t_span=[T0, T], y0=[Y0], t_eval=t_array, max_step = h, min_step = h)
    except:
        print('')

    ansFEM, arrFEM = ForEuler(dY_dT, Y0, T0, T, iterations)
    ansRK2, arrRK2 = RK2(dY_dT, Y0, T0, T, iterations)
    ansRK4, arrRK4 = RK4(dY_dT, Y0, T0, T, iterations)
    ansSol = soln(Y0, T0, T)

    #errFEM = errorFinder(ansSol, arrFEM)
    #errRK2 = errorFinder(ansSol, arrRK2)    
    #errRK4 = errorFinder(ansSol, arrRK4)
    #errSolveIVP = errorFinder(ansSol, solveIVP.y[:, -1])

    if showsoln:
        print(f"\n\n--------------------Solutions after {iterations} iterations--------------------")

        print(f'exact solution is  {ansSol:.10f}')
        print(f'Answer by FEM  is  {ansFEM:.10f} with error {abs(ansFEM -ansSol):.10f}')
        print(f'Answer by RK2  is  {ansRK2:.10f} with error {abs(ansRK2 -ansSol):.10f}')
        print(f'Answer by RK4  is  {ansRK4:.10f} with error {abs(ansRK4 -ansSol):.10f}')
        print(f'Answer by SIVP is  {(solveIVP.y[:, -1])[0]:.10f} with error {abs((ansSol - solveIVP.y[:, -1])[0]):.10f}')

    if showgraph:

        timearr = np.linspace(T0, T, iterations + 1)
        solnarr = [soln(Y0, T0, t) for t in timearr]  # Exact solution values
        
        #solnarr = []
        #print(solnarr)


        plt.figure(figsize=(10, 6))
        #plt.axhline(y= ansSol, color="red", label = "Actual Solution")
        plt.plot(timearr, solnarr, 'o', label="Exact Solution", color="#B8860B", markersize=6)
        plt.plot(timearr, arrFEM, label="Forward Euler Method", color="blue", linestyle="--")
        plt.plot(timearr, arrRK2, label="RK2 Method", color="green", linestyle="-.")
        plt.plot(timearr, arrRK4, label="RK4 Method", color="red", linestyle="-")
        plt.plot(timearr, solveIVP.y[0], label="solve_ivp", color="black", linestyle="dashed")

        #plt.ylim(min(solnarr) - 0.3, max(solnarr) + 0.3) 

        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Solutions ", fontsize=12)
        plt.grid(True)

        plt.legend()
        plt.show()

        print("errors are")
        print(ansSol - ansFEM, ansSol - ansRK2, ansSol - ansRK4, (ansSol - solveIVP.y[:, -1])[0])
    return abs(ansSol - ansFEM), abs(ansSol - ansRK2), abs(ansSol - ansRK4), abs((ansSol - solveIVP.y[:, -1])[0])

def OrderOfConvergence(dY_dT, Y0, T0, T):

    #iterationsList = [10, 100, 1000]
    iterationsList = [20, 40, 60, 80, 100, 120]
    h = [(T-T0)/i for i in iterationsList]

    arrFEM, arrRK2, arrRK4, arrSolveIVP = [], [], [], []

    for i in iterationsList:
        errFEM, errRK2, errRK4, errSolveIVP = main(DE, Y0, T0, T, i, False, False)
        arrFEM.append(errFEM)
        arrRK2.append(errRK2)
        arrRK4.append(errRK4)
        arrSolveIVP.append(errSolveIVP)

    #print(arrFEM, arrRK2, arrRK4, arrSolveIVP)
    
    def orderfinder(estimates, h):
        
        p = []
        #print(f'erros are {estimates}')
        for i in range(len(estimates) - 1):
            pAppend = math.log(estimates[i]/estimates[i+1])/math.log(h[i]/h[i+1])
            p.append(pAppend)

        return p

    def print_orders(name, orders):
        formatted_orders = ", ".join(f"{order:.4f}" for order in orders)
        print(f"{name:<9} has orders   {formatted_orders:<25}{orders[-1]:.4f}")

    
    print("\n\n------------------------Orders of convergence------------------------")
    print("Iterations             20-40   40-60   60-80   80-100  100-120")
    print_orders("FEM", orderfinder(arrFEM, h))
    print_orders("RK2", orderfinder(arrRK2 ,h))
    print_orders("RK4", orderfinder(arrRK4 ,h))
    print_orders("SolveIVP", orderfinder(arrSolveIVP ,h))

Y0 = 1
T0 = 0
T = 10
iterations = 100

main(DE, Y0, T0, T, iterations, showgraph=1, showsoln=1)
OrderOfConvergence(DE, Y0, T0, T)