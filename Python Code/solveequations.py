import sympy as sp

# Unknown forces
f_RA, f_LA, f_RL, f_LL = sp.symbols("f_RA f_LA f_RL f_LL")

# Direction vector components
aRAx, aRAy, aRAz = sp.symbols("aRAx aRAy aRAz")
aLAx, aLAy, aLAz = sp.symbols("aLAx aLAy aLAz")
aRLx, aRLy, aRLz = sp.symbols("aRLx aRLy aRLz")
aLLx, aLLy, aLLz = sp.symbols("aLLx aLLy aLLz")

# Lever arm components
dRAx, dRAy, dRAz = sp.symbols("dRAx dRAy dRAz")
dLAx, dLAy, dLAz = sp.symbols("dLAx dLAy dLAz")
dRLx, dRLy, dRLz = sp.symbols("dRLx dRLy dRLz")
dLLx, dLLy, dLLz = sp.symbols("dLLx dLLy dLLz")
dMMx, dMMy, dMMz = sp.symbols("dMMx dMMy dMMz")

# Contact Points
RHx, RHy, RHz = sp.symbols("RHx RHy RHz")
LHx, LHy, LHz = sp.symbols("LHx LHy LHz")
RFx, RFy, RFz = sp.symbols("RFx RFy RFz")
LFx, LFy, LFz = sp.symbols("LFx LFy LFz")

# COM
CMx, CMy, CMz = sp.symbols("CMx CMy CMz")

# Gravity components
gx, gy, gz = sp.symbols("gx gy gz")

# Force equilibrium equations
eq1 = sp.Eq(f_RA*aRAx + f_LA*aLAx + f_RL*aRLx + f_LL*aLLx + gx, 0)
eq2 = sp.Eq(f_RA*aRAy + f_LA*aLAy + f_RL*aRLy + f_LL*aLLy + gy, 0)
eq3 = sp.Eq(f_RA*aRAz + f_LA*aLAz + f_RL*aRLz + f_LL*aLLz + gz, 0)

# Moment equilibrium equations (cross products expanded)
eq4 = sp.Eq(f_RA*(dRAy*aRAz - dRAz*aRAy) +
            f_LA*(dLAy*aLAz - dLAz*aLAy) +
            f_RL*(dRLy*aRLz - dRLz*aRLy) +
            f_LL*(dLLy*aLLz - dLLz*aLLy) + (dMMy*gz-dMMz*gy), 0)


F_init = sp.sqrt((f_RA*aRAx) ** 2  + (f_RA*aRAy) ** 2  + (f_RA*aRAz) ** 2)+ sp.sqrt( + (f_LA*aLAx) ** 2 + (f_LA*aLAy) ** 2 + (f_LA*aLAz) ** 2)

F_sub = F_init.subs({
    f_RA: (aLAx*aLLy*aRLy*dLLz*gz - aLAx*aLLy*aRLy*dRLz*gz - aLAx*aLLy*aRLz*dLLz*gy - aLAx*aLLy*aRLz*dMMy*gz + aLAx*aLLy*aRLz*dMMz*gy + aLAx*aLLy*aRLz*dRLy*gz - aLAx*aLLz*aRLy*dLLy*gz + aLAx*aLLz*aRLy*dMMy*gz - aLAx*aLLz*aRLy*dMMz*gy + aLAx*aLLz*aRLy*dRLz*gy + aLAx*aLLz*aRLz*dLLy*gy - aLAx*aLLz*aRLz*dRLy*gy - aLAy*aLLx*aRLy*dLAz*gz + aLAy*aLLx*aRLy*dRLz*gz + aLAy*aLLx*aRLz*dLAz*gy + aLAy*aLLx*aRLz*dMMy*gz - aLAy*aLLx*aRLz*dMMz*gy - aLAy*aLLx*aRLz*dRLy*gz + aLAy*aLLy*aRLx*dLAz*gz - aLAy*aLLy*aRLx*dLLz*gz - aLAy*aLLy*aRLz*dLAz*gx + aLAy*aLLy*aRLz*dLLz*gx - aLAy*aLLz*aRLx*dLAz*gy + aLAy*aLLz*aRLx*dLLy*gz - aLAy*aLLz*aRLx*dMMy*gz + aLAy*aLLz*aRLx*dMMz*gy + aLAy*aLLz*aRLy*dLAz*gx - aLAy*aLLz*aRLy*dRLz*gx - aLAy*aLLz*aRLz*dLLy*gx + aLAy*aLLz*aRLz*dRLy*gx + aLAz*aLLx*aRLy*dLAy*gz - aLAz*aLLx*aRLy*dMMy*gz + aLAz*aLLx*aRLy*dMMz*gy - aLAz*aLLx*aRLy*dRLz*gy - aLAz*aLLx*aRLz*dLAy*gy + aLAz*aLLx*aRLz*dRLy*gy - aLAz*aLLy*aRLx*dLAy*gz + aLAz*aLLy*aRLx*dLLz*gy + aLAz*aLLy*aRLx*dMMy*gz - aLAz*aLLy*aRLx*dMMz*gy - aLAz*aLLy*aRLy*dLLz*gx + aLAz*aLLy*aRLy*dRLz*gx + aLAz*aLLy*aRLz*dLAy*gx - aLAz*aLLy*aRLz*dRLy*gx + aLAz*aLLz*aRLx*dLAy*gy - aLAz*aLLz*aRLx*dLLy*gy - aLAz*aLLz*aRLy*dLAy*gx + aLAz*aLLz*aRLy*dLLy*gx)/(aLAx*aLLy*aRAy*aRLz*dLLz - aLAx*aLLy*aRAy*aRLz*dRAz - aLAx*aLLy*aRAz*aRLy*dLLz + aLAx*aLLy*aRAz*aRLy*dRLz + aLAx*aLLy*aRAz*aRLz*dRAy - aLAx*aLLy*aRAz*aRLz*dRLy + aLAx*aLLz*aRAy*aRLy*dRAz - aLAx*aLLz*aRAy*aRLy*dRLz - aLAx*aLLz*aRAy*aRLz*dLLy + aLAx*aLLz*aRAy*aRLz*dRLy + aLAx*aLLz*aRAz*aRLy*dLLy - aLAx*aLLz*aRAz*aRLy*dRAy - aLAy*aLLx*aRAy*aRLz*dLAz + aLAy*aLLx*aRAy*aRLz*dRAz + aLAy*aLLx*aRAz*aRLy*dLAz - aLAy*aLLx*aRAz*aRLy*dRLz - aLAy*aLLx*aRAz*aRLz*dRAy + aLAy*aLLx*aRAz*aRLz*dRLy + aLAy*aLLy*aRAx*aRLz*dLAz - aLAy*aLLy*aRAx*aRLz*dLLz - aLAy*aLLy*aRAz*aRLx*dLAz + aLAy*aLLy*aRAz*aRLx*dLLz - aLAy*aLLz*aRAx*aRLy*dLAz + aLAy*aLLz*aRAx*aRLy*dRLz + aLAy*aLLz*aRAx*aRLz*dLLy - aLAy*aLLz*aRAx*aRLz*dRLy + aLAy*aLLz*aRAy*aRLx*dLAz - aLAy*aLLz*aRAy*aRLx*dRAz - aLAy*aLLz*aRAz*aRLx*dLLy + aLAy*aLLz*aRAz*aRLx*dRAy - aLAz*aLLx*aRAy*aRLy*dRAz + aLAz*aLLx*aRAy*aRLy*dRLz + aLAz*aLLx*aRAy*aRLz*dLAy - aLAz*aLLx*aRAy*aRLz*dRLy - aLAz*aLLx*aRAz*aRLy*dLAy + aLAz*aLLx*aRAz*aRLy*dRAy + aLAz*aLLy*aRAx*aRLy*dLLz - aLAz*aLLy*aRAx*aRLy*dRLz - aLAz*aLLy*aRAx*aRLz*dLAy + aLAz*aLLy*aRAx*aRLz*dRLy - aLAz*aLLy*aRAy*aRLx*dLLz + aLAz*aLLy*aRAy*aRLx*dRAz + aLAz*aLLy*aRAz*aRLx*dLAy - aLAz*aLLy*aRAz*aRLx*dRAy + aLAz*aLLz*aRAx*aRLy*dLAy - aLAz*aLLz*aRAx*aRLy*dLLy - aLAz*aLLz*aRAy*aRLx*dLAy + aLAz*aLLz*aRAy*aRLx*dLLy),
    f_LA: (aLLx*aRAy*aRLy*dRAz*gz - aLLx*aRAy*aRLy*dRLz*gz - aLLx*aRAy*aRLz*dMMy*gz + aLLx*aRAy*aRLz*dMMz*gy - aLLx*aRAy*aRLz*dRAz*gy + aLLx*aRAy*aRLz*dRLy*gz + aLLx*aRAz*aRLy*dMMy*gz - aLLx*aRAz*aRLy*dMMz*gy - aLLx*aRAz*aRLy*dRAy*gz + aLLx*aRAz*aRLy*dRLz*gy + aLLx*aRAz*aRLz*dRAy*gy - aLLx*aRAz*aRLz*dRLy*gy - aLLy*aRAx*aRLy*dLLz*gz + aLLy*aRAx*aRLy*dRLz*gz + aLLy*aRAx*aRLz*dLLz*gy + aLLy*aRAx*aRLz*dMMy*gz - aLLy*aRAx*aRLz*dMMz*gy - aLLy*aRAx*aRLz*dRLy*gz + aLLy*aRAy*aRLx*dLLz*gz - aLLy*aRAy*aRLx*dRAz*gz - aLLy*aRAy*aRLz*dLLz*gx + aLLy*aRAy*aRLz*dRAz*gx - aLLy*aRAz*aRLx*dLLz*gy - aLLy*aRAz*aRLx*dMMy*gz + aLLy*aRAz*aRLx*dMMz*gy + aLLy*aRAz*aRLx*dRAy*gz + aLLy*aRAz*aRLy*dLLz*gx - aLLy*aRAz*aRLy*dRLz*gx - aLLy*aRAz*aRLz*dRAy*gx + aLLy*aRAz*aRLz*dRLy*gx + aLLz*aRAx*aRLy*dLLy*gz - aLLz*aRAx*aRLy*dMMy*gz + aLLz*aRAx*aRLy*dMMz*gy - aLLz*aRAx*aRLy*dRLz*gy - aLLz*aRAx*aRLz*dLLy*gy + aLLz*aRAx*aRLz*dRLy*gy - aLLz*aRAy*aRLx*dLLy*gz + aLLz*aRAy*aRLx*dMMy*gz - aLLz*aRAy*aRLx*dMMz*gy + aLLz*aRAy*aRLx*dRAz*gy - aLLz*aRAy*aRLy*dRAz*gx + aLLz*aRAy*aRLy*dRLz*gx + aLLz*aRAy*aRLz*dLLy*gx - aLLz*aRAy*aRLz*dRLy*gx + aLLz*aRAz*aRLx*dLLy*gy - aLLz*aRAz*aRLx*dRAy*gy - aLLz*aRAz*aRLy*dLLy*gx + aLLz*aRAz*aRLy*dRAy*gx)/(aLAx*aLLy*aRAy*aRLz*dLLz - aLAx*aLLy*aRAy*aRLz*dRAz - aLAx*aLLy*aRAz*aRLy*dLLz + aLAx*aLLy*aRAz*aRLy*dRLz + aLAx*aLLy*aRAz*aRLz*dRAy - aLAx*aLLy*aRAz*aRLz*dRLy + aLAx*aLLz*aRAy*aRLy*dRAz - aLAx*aLLz*aRAy*aRLy*dRLz - aLAx*aLLz*aRAy*aRLz*dLLy + aLAx*aLLz*aRAy*aRLz*dRLy + aLAx*aLLz*aRAz*aRLy*dLLy - aLAx*aLLz*aRAz*aRLy*dRAy - aLAy*aLLx*aRAy*aRLz*dLAz + aLAy*aLLx*aRAy*aRLz*dRAz + aLAy*aLLx*aRAz*aRLy*dLAz - aLAy*aLLx*aRAz*aRLy*dRLz - aLAy*aLLx*aRAz*aRLz*dRAy + aLAy*aLLx*aRAz*aRLz*dRLy + aLAy*aLLy*aRAx*aRLz*dLAz - aLAy*aLLy*aRAx*aRLz*dLLz - aLAy*aLLy*aRAz*aRLx*dLAz + aLAy*aLLy*aRAz*aRLx*dLLz - aLAy*aLLz*aRAx*aRLy*dLAz + aLAy*aLLz*aRAx*aRLy*dRLz + aLAy*aLLz*aRAx*aRLz*dLLy - aLAy*aLLz*aRAx*aRLz*dRLy + aLAy*aLLz*aRAy*aRLx*dLAz - aLAy*aLLz*aRAy*aRLx*dRAz - aLAy*aLLz*aRAz*aRLx*dLLy + aLAy*aLLz*aRAz*aRLx*dRAy - aLAz*aLLx*aRAy*aRLy*dRAz + aLAz*aLLx*aRAy*aRLy*dRLz + aLAz*aLLx*aRAy*aRLz*dLAy - aLAz*aLLx*aRAy*aRLz*dRLy - aLAz*aLLx*aRAz*aRLy*dLAy + aLAz*aLLx*aRAz*aRLy*dRAy + aLAz*aLLy*aRAx*aRLy*dLLz - aLAz*aLLy*aRAx*aRLy*dRLz - aLAz*aLLy*aRAx*aRLz*dLAy + aLAz*aLLy*aRAx*aRLz*dRLy - aLAz*aLLy*aRAy*aRLx*dLLz + aLAz*aLLy*aRAy*aRLx*dRAz + aLAz*aLLy*aRAz*aRLx*dLAy - aLAz*aLLy*aRAz*aRLx*dRAy + aLAz*aLLz*aRAx*aRLy*dLAy - aLAz*aLLz*aRAx*aRLy*dLLy - aLAz*aLLz*aRAy*aRLx*dLAy + aLAz*aLLz*aRAy*aRLx*dLLy),
})

F = F_sub.subs({
    aRAx: RHx - CMx,
    aRAy: RHy - CMy,
    aRAz: RHz - CMz,
    aLAx: LHx - CMx,
    aLAy: LHy - CMy,
    aLAz: LHz - CMz,
    aRLx: RFx - CMx,
    aRLy: RFy - CMy,
    aRLz: RFz - CMz,
    aLLx: LFx - CMx,
    aLLy: LFy - CMy,
    aLLz: LFz - CMz,
})

# Partial derivatives
df_dx = sp.diff(F, CMx)
df_dy = sp.diff(F, CMy)
df_dz = sp.diff(F, CMz)

print("F'(CMx) = ")
print(df_dx)
print("F'(CMy) = ")
print(df_dy)
print("F'(CMz) = ")
print(df_dz)

# Solve system
#solution = sp.solve([eq1, eq2, eq3, eq4], (f_RA, f_LA, f_RL, f_LL), dict=True)
#print(solution)

