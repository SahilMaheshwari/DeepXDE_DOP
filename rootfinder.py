import sympy as sp
import math

# Define the symbolic variable
x = sp.Symbol('x')

# Define symbolic parameters
I, r_1, m_1, r_2, r_3, m_2, m_3, c1, h_2, h_3 = sp.symbols('I r_1 m_1 r_2 r_3 m_2 m_3 c1 h_2 h_3')

# Define the equation
eq = (I * r_1 * x / (x * r_1 - sp.exp(1))) - m_1 - (
    (2 * x * r_2 * r_3 - r_2 * m_3 - r_3 * m_2 - r_2 * r_3 * x * (m_3 * h_3 + m_2 * h_2)) / 
    (c1 * (1 + h_2 * r_2 * x) * (1 + h_3 * r_3 * x))
)

# Assign numerical values to parameters
params = {
    I: 20000, r_1: 8, m_1: 0.03, 
    r_2: 0.02, h_2: 0.5, m_2: 0.12, c1: 0.05, 
    r_3: 0.03, h_3: 0.1, m_3: 0.04
}

params = {
    I: 50000,   # Increase I to ensure n0 > 0
    r_1: 5.0,   # Increase r_1
    m_1: 0,  
    r_2: 1.2,   # Increase r_2
    h_2: 0.002, # Reduce h_2
    m_2: 0.0001, # Reduce m_2
    c1: 0.0002,  # Reduce c1
    r_3: 1.5,   # Increase r_3
    h_3: 0.002, # Reduce h_3
    m_3: 0.0001  # Reduce m_3
}


# Substitute numerical values
eq_numeric = eq.subs(params)

# Try multiple initial guesses
initial_guesses = [-10, -1, 0.1, 1, 10, 100, 10000]  # Different guesses to find different roots
solutions = set()  # Use a set to store unique solutions

for guess in initial_guesses:
    try:
        sol = sp.nsolve(eq_numeric, x, guess)
        solutions.add(float(sol))  # Convert to float to avoid duplicate symbolic values
    except:
        pass  # Ignore errors for invalid guesses

# Compute steady states for each found solution
print("STEADY STATES:")

for n1_ss in sorted(solutions):
    n0_ss = params[I] / (n1_ss * params[r_1] - math.e)
    n2_ss = (1 / params[c1]) * (((n1_ss * params[r_3]) / (1 + params[h_3] * params[r_3] * n1_ss)) - params[m_3])
    n3_ss = (1 / params[c1]) * (((n1_ss * params[r_2]) / (1 + params[h_2] * params[r_2] * n1_ss)) - params[m_2])

    print(f"\nn0: {n0_ss}")
    print(f"n1: {n1_ss}")
    print(f"n2: {n2_ss}")
    print(f"n3: {n3_ss}")

