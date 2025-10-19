import sympy as sp

# Define variables
F_x1, F_x2, F_x3, F_x4 = sp.symbols('F_x1 F_x2 F_x3 F_x4')
F_a,F_c,F_r,F_t,F_b,F_d,F_s,F_u,F_p = sp.symbols('F_a F_c F_r F_t F_b F_d F_s F_u F_p')
F_dfy,F_dfz,F_dgy,F_dgz,F_dhy,F_dhz,F_djy,F_djz = sp.symbols('F_dfy F_dfz F_dgy F_dgz F_dhy F_dhz F_djy F_djz')

# Equations
eq1 = sp.Eq(F_x1 + F_x2 + F_x3 + F_x4, 0)
eq2 = sp.Eq(F_a*F_x1 + F_c*F_x2 + F_r*F_x3 + F_t*F_x4, 0)
eq3 = sp.Eq(F_b*F_x1 + F_d*F_x2 + F_s*F_x3 + F_u*F_x4, F_p)
eq4 = sp.Eq((F_dfy*F_b - F_dfz*F_a)*F_x1 + (F_dgy*F_d - F_dgz*F_c)*F_x2 + (F_dhy*F_s - F_dhz*F_r)*F_x3 + (F_djy*F_u - F_djz*F_t)*F_x4, 0)

# Solve system
solution = sp.solve([eq1, eq2, eq3, eq4], (F_x1, F_x2, F_x3, F_x4), dict=True)[0]
print(solution)

# Example: substitute numeric values
values = {
    F_a:1, F_c:2, F_r:3, F_t:4,
    F_b:5, F_d:6, F_s:7, F_u:15,
    F_dfy:12, F_dfz:2, F_dgy:3, F_dgz:4,
    F_dhy:5, F_dhz:6, F_djy:7, F_djz:8,
    F_p:10
}

numeric_solution = {var: solution[var].subs(values).evalf() for var in solution}
print(numeric_solution)
