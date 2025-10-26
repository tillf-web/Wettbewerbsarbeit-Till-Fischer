import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
import sympy as sp
import math
import cvxpy as cp


# Configuration
model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
output_dir = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/output2/"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Geometry"
pose_name = "static_climb"
total_mass = 65  # kg
gravity = 9.81
weight = total_mass * gravity
force_each = weight / 4

# Limit angles and strength config (these might be truncated in display but are unchanged from your file)
shoulder_flexion_max = 120 * np.pi / 180
shoulder_flexion_min = -60 * np.pi / 180
elbow_flexion_max = 150 * np.pi / 180
elbow_flexion_min = 0 * np.pi / 180
hip_flexion_max = 120 * np.pi / 180
hip_flexion_min = -30 * np.pi / 180
knee_flexion_max = 160 * np.pi / 180
knee_flexion_min = 0 * np.pi / 180



# === STEP 1: Load and Prepare Model ===
model = osim.Model(model_path)

# Optional: unlock defaults before initSystem
for i in range(model.getCoordinateSet().getSize()):
    coord = model.getCoordinateSet().get(i)
    coord.setDefaultLocked(False)

# Initialize
state = model.initSystem()
# marker offsets etc.
hold_names = ["hand_l", "hand_r", "toes_l", "toes_r"]



hold_positions = {}


for comp in model.getComponentsList():
    if isinstance(comp, osim.PhysicalOffsetFrame):
        name = comp.getName()
        if name.endswith("_off"):
            base_name = name[:-4]  # strip "_off"
            if base_name in hold_names:
                location = comp.getPositionInGround(state)
                hold_positions[base_name] = [location.get(i) for i in range(3)]
                hold_positions[base_name+"_tip"] = [location.get(i) for i in range(3)]


hold_positions["hand_l_tip"][0]+=0.02
hold_positions["hand_r_tip"][0]+=0.02

# Marker / body mapping
body_map = {
    "hand_l": "hand_l/hand_l_geom_16",
    "hand_r": "hand_r/hand_r_geom_16",
    "toes_l": "toes_l/toes_l_geom_1",
    "toes_r": "toes_r/toes_r_geom_1",
}

if os.path.exists(os.path.join(output_dir, "summary_pose.mot")):
    os.remove(os.path.join(output_dir, "summary_pose.mot"))

# Unknown forces (scalars, magnitudes along each limb's line of action)
f_RA, f_LA, f_RL, f_LL = sp.symbols("f_RA f_LA f_RL f_LL")

# Direction vectors (COM to contact etc.)
aRAx, aRAy, aRAz = sp.symbols("aRAx aRAy aRAz")
aLAx, aLAy, aLAz = sp.symbols("aLAx aLAy aLAz")
aRLx, aRLy, aRLz = sp.symbols("aRLx aRLy aRLz")
aLLx, aLLy, aLLz = sp.symbols("aLLx aLLy aLLz")

# Moment arm vectors (contact point relative to moment center)
dRAx, dRAy, dRAz = sp.symbols("dRAx dRAy dRAz")
dLAx, dLAy, dLAz = sp.symbols("dLAx dLAy dLAz")
dRLx, dRLy, dRLz = sp.symbols("dRLx dRLy dRLz")
dLLx, dLLy, dLLz = sp.symbols("dLLx dLLy dLLz")

# COM relative to moment center
dMMx, dMMy, dMMz = sp.symbols("dMMx dMMy dMMz")

# Marker positions in world
RHx, RHy, RHz = sp.symbols("RHx RHy RHz")
LHx, LHy, LHz = sp.symbols("LHx LHy LHz")
RFx, RFy, RFz = sp.symbols("RFx RFy RFz")
LFx, LFy, LFz = sp.symbols("LFx LFy LFz")

# COM position
CMx, CMy, CMz = sp.symbols("CMx CMy CMz")

# Gravity components
gx, gy, gz = sp.symbols("gx gy gz")

# Force equilibrium equations
eq1 = sp.Eq(f_RA * aRAx + f_LA * aLAx + f_RL * aRLx + f_LL * aLLx + gx, 0)
eq2 = sp.Eq(f_RA * aRAy + f_LA * aLAy + f_RL * aRLy + f_LL * aLLy + gy, 0)
eq3 = sp.Eq(f_RA * aRAz + f_LA * aLAz + f_RL * aRLz + f_LL * aLLz + gz, 0)

# Moment equilibrium equation about MOMENTCENT (you derived full expansion already)
eq4 = sp.Eq(
    f_RA * (dRAy * aRAz - dRAz * aRAy)
    + f_LA * (dLAy * aLAz - dLAz * aLAy)
    + f_RL * (dRLy * aRLz - dRLz * aRLy)
    + f_LL * (dLLy * aLLz - dLLz * aLLy)
    + (dMMy * gz - dMMz * gy),
    0,
)

# Total “effort” magnitude F (sum of squared limb force vectors, then sqrt)
# (This is the long symbolic expression in your file - kept as-is here.
# Display truncation may show "..." but the actual line content is unchanged.)
F_init = sp.sqrt(
    (f_RA * aRAx) ** 2
    + (f_RA * aRAy) ** 2
    + (f_RA * aRAz) ** 2
    + (f_RL * aRLx) ** 2
    + (f_RL * aRLy) ** 2
    + (f_RL * aRLz) ** 2
    + (f_LL * aLLx) ** 2
    + (f_LL * aLLy) ** 2
    + (f_LL * aLLz) ** 2
    + (f_LA * aLAx) ** 2
    + (f_LA * aLAy) ** 2
    + (f_LA * aLAz) ** 2
)

# F_sub was created in your original file by analytically solving eq1..eq4 for f_LA,f_RA,f_RL,f_LL,
# and substituting that into F_init. The actual giant expression is still present in your file
# even if it's visually truncated here.
F_sub = F_init.subs({
    f_RA: (aLAx*aLLy*aRLy*dLLz*gz - aLAx*aLLy*aRLy*dRLz*gz - aLAx*aLLy*aRLz*dLLy*gz + aLAx*aLLy*aRLz*dRLy*gz + aLAx*aLLz*aRLy*dLLy*gz - aLAx*aLLz*aRLy*dRLy*gz - aLAx*aLLz*aRLz*dLLy*gy + aLAx*aLLz*aRLz*dRLy*gy - aLAx*aRLy*aRLz*dLLy*gy + aLAx*aRLy*aRLz*dRLy*gy - aLAy*aLLx*aRLy*dLLz*gz + aLAy*aLLx*aRLy*dRLz*gz + aLAy*aLLx*aRLz*dLLy*gz - aLAy*aLLx*aRLz*dRLy*gz - aLAy*aLLz*aRLx*dLLy*gz + aLAy*aLLz*aRLx*dRLy*gz + aLAy*aLLz*aRLz*dLLx*gy - aLAy*aLLz*aRLz*dRLx*gy + aLAy*aRLx*aRLz*dLLy*gy - aLAy*aRLx*aRLz*dRLy*gy + aLAz*aLLx*aRLy*dLLy*gz - aLAz*aLLx*aRLy*dRLy*gz - aLAz*aLLx*aRLz*dLLy*gy + aLAz*aLLx*aRLz*dRLy*gy + aLAz*aLLy*aRLx*dLLy*gz - aLAz*aLLy*aRLx*dRLy*gz - aLAz*aLLy*aRLz*dLLx*gy + aLAz*aLLy*aRLz*dRLx*gy - aLAz*aRLx*aRLy*dLLy*gy + aLAz*aRLx*aRLy*dRLy*gy + aLLx*aRLy*aRLz*dLLy*gy - aLLx*aRLy*aRLz*dRLy*gy - aLLy*aRLx*aRLz*dLLy*gy + aLLy*aRLx*aRLz*dRLy*gy + aLLy*aRLy*aRLz*dLLx*gy - aLLy*aRLy*aRLz*dRLx*gy - aRAx*aLAy*aLLy*aRLy*dLLz*gz + aRAx*aLAy*aLLy*aRLy*dRLz*gz + aRAx*aLAy*aLLy*aRLz*dLLy*gz - aRAx*aLAy*aLLy*aRLz*dRLy*gz - aRAx*aLAy*aLLz*aRLy*dLLy*gz + aRAx*aLAy*aLLz*aRLy*dRLy*gz + aRAx*aLAy*aLLz*aRLz*dLLy*gy - aRAx*aLAy*aLLz*aRLz*dRLy*gy + aRAx*aLAy*aRLy*aRLz*dLLy*gy - aRAx*aLAy*aRLy*aRLz*dRLy*gy + aRAx*aLAz*aLLy*aRLy*dLLy*gz - aRAx*aLAz*aLLy*aRLy*dRLy*gz - aRAx*aLAz*aLLy*aRLz*dLLy*gy + aRAx*aLAz*aLLy*aRLz*dRLy*gy + aRAx*aLAz*aLLz*aRLy*dLLy*gy - aRAx*aLAz*aLLz*aRLy*dRLy*gy - aRAx*aLAz*aRLx*aRLy*dLLy*gy + aRAx*aLAz*aRLx*aRLy*dRLy*gy - aRAx*aLLy*aRLy*aRLz*dLLy*gy + aRAx*aLLy*aRLy*aRLz*dRLy*gy + aRAy*aLAx*aLLy*aRLy*dLLz*gz - aRAy*aLAx*aLLy*aRLy*dRLz*gz - aRAy*aLAx*aLLy*aRLz*dLLy*gz + aRAy*aLAx*aLLy*aRLz*dRLy*gz + aRAy*aLAx*aLLz*aRLy*dLLy*gz - aRAy*aLAx*aLLz*aRLy*dRLy*gz - aRAy*aLAx*aLLz*aRLz*dLLy*gy + aRAy*aLAx*aLLz*aRLz*dRLy*gy - aRAy*aLAx*aRLy*aRLz*dLLy*gy + aRAy*aLAx*aRLy*aRLz*dRLy*gy - aRAy*aLAz*aLLx*aRLy*dLLy*gz + aRAy*aLAz*aLLx*aRLy*dRLy*gz + aRAy*aLAz*aLLx*aRLz*dLLy*gy - aRAy*aLAz*aLLx*aRLz*dRLy*gy - aRAy*aLAz*aLLy*aRLx*dLLy*gz + aRAy*aLAz*aLLy*aRLx*dRLy*gz + aRAy*aLAz*aRLx*aRLy*dLLy*gy - aRAy*aLAz*aRLx*aRLy*dRLy*gy + aRAy*aLLx*aRLy*aRLz*dLLy*gy - aRAy*aLLx*aRLy*aRLz*dRLy*gy - aRAz*aLAx*aLLy*aRLy*dLLy*gz + aRAz*aLAx*aLLy*aRLy*dRLy*gz + aRAz*aLAx*aLLy*aRLz*dLLy*gy - aRAz*aLAx*aLLy*aRLz*dRLy*gy - aRAz*aLAx*aLLz*aRLy*dLLy*gy + aRAz*aLAx*aLLz*aRLy*dRLy*gy + aRAz*aLAx*aRLx*aRLy*dLLy*gy - aRAz*aLAx*aRLx*aRLy*dRLy*gy + aRAz*aLAy*aLLx*aRLy*dLLy*gz - aRAz*aLAy*aLLx*aRLy*dRLy*gz - aRAz*aLAy*aLLx*aRLz*dLLy*gy + aRAz*aLAy*aLLx*aRLz*dRLy*gy + aRAz*aLAy*aLLy*aRLx*dLLy*gz - aRAz*aLAy*aLLy*aRLx*dRLy*gz - aRAz*aLAy*aRLx*aRLy*dLLy*gy + aRAz*aLAy*aRLx*aRLy*dRLy*gy - aRAz*aLLx*aRLy*aRLz*dLLy*gy + aRAz*aLLx*aRLy*aRLz*dRLy*gy + aLLx*aRLy*aRLz*dLAy*gy - aLLx*aRLy*aRLz*dRAy*gy - aLLy*aRLx*aRLz*dLAy*gy + aLLy*aRLx*aRLz*dRAy*gy + aLLy*aRLy*aRLz*dLAx*gy - aLLy*aRLy*aRLz*dRAx*gy - aLAx*aLLy*aRLy*dLLz*gz + aLAx*aLLy*aRLy*dRLz*gz + aLAx*aLLy*aRLz*dLLy*gz - aLAx*aLLy*aRLz*dRLy*gz - aLAx*aLLz*aRLy*dLLy*gz + aLAx*aLLz*aRLy*dRLy*gz + aLAx*aLLz*aRLz*dLLy*gy - aLAx*aLLz*aRLz*dRLy*gy + aLAx*aRLy*aRLz*dLLy*gy - aLAx*aRLy*aRLz*dRLy*gy + aLAy*aLLx*aRLy*dLLy*gz - aLAy*aLLx*aRLy*dRLy*gz - aLAy*aLLx*aRLz*dLLy*gy + aLAy*aLLx*aRLz*dRLy*gy + aLAy*aLLy*aRLx*dLLy*gz - aLAy*aLLy*aRLx*dRLy*gz - aLAy*aRLx*aRLy*dLLy*gy + aLAy*aRLx*aRLy*dRLy*gy + aLAz*aLLx*aRLy*dLLy*gy - aLAz*aLLx*aRLy*dRLy*gy - aLAz*aLLy*aRLx*dLLy*gy + aLAz*aLLy*aRLx*dRLy*gy + aLAz*aRLx*aRLy*dLLy*gy - aLAz*aRLx*aRLy*dRLy*gy - aLLx*aRLy*aRLz*dLAy*gy + aLLx*aRLy*aRLz*dRAy*gy + aLLy*aRLx*aRLz*dLAy*gy - aLLy*aRLx*aRLz*dRAy*gy - aLLy*aRLy*aRLz*dLAx*gy + aLLy*aRLy*aRLz*dRAx*gy)/(aLAx*aLLy*aRLy*aRAz - aLAx*aLLy*aRLy*aRLz + aLAx*aLLy*aRLy*aLLz - aLAx*aLLy*aRLz*aRAy + aLAx*aLLy*aRLz*aRLy - aLAx*aLLy*aRLz*aLLy - aLAx*aLLz*aRLy*aRAy + aLAx*aLLz*aRLy*aRLy - aLAx*aLLz*aRLy*aLLy + aLAx*aLLz*aRLz*aRAy - aLAx*aLLz*aRLz*aRLy + aLAx*aLLz*aRLz*aLLy + aLAx*aRLy*aRLz*aRAy - aLAx*aRLy*aRLz*aRLy + aLAx*aRLy*aRLz*aLLy - aLAy*aLLx*aRLy*aRAz + aLAy*aLLx*aRLy*aRLz - aLAy*aLLx*aRLy*aLLz + aLAy*aLLx*aRLz*aRAx - aLAy*aLLx*aRLz*aRLx + aLAy*aLLx*aRLz*aLLx + aLAy*aLLz*aRLx*aRAy - aLAy*aLLz*aRLx*aRLy + aLAy*aLLz*aRLx*aLLy - aLAy*aLLz*aRLz*aRAx + aLAy*aLLz*aRLz*aRLx - aLAy*aLLz*aRLz*aLLx - aLAy*aRLx*aRLz*aRAx + aLAy*aRLx*aRLz*aRLx - aLAy*aRLx*aRLz*aLLx + aLAz*aLLx*aRLy*aRAy - aLAz*aLLx*aRLy*aRLy + aLAz*aLLx*aRLy*aLLy - aLAz*aLLx*aRLz*aRAy + aLAz*aLLx*aRLz*aRLy - aLAz*aLLx*aRLz*aLLy - aLAz*aLLy*aRLx*aRAy + aLAz*aLLy*aRLx*aRLy - aLAz*aLLy*aRLx*aLLy + aLAz*aLLy*aRLz*aRAx - aLAz*aLLy*aRLz*aRLx + aLAz*aLLy*aRLz*aLLx + aLAz*aRLx*aRLy*aRAx - aLAz*aRLx*aRLy*aRLx + aLAz*aRLx*aRLy*aLLx - aLLx*aRLy*aRLz*aRAy + aLLx*aRLy*aRLz*aRLy - aLLx*aRLy*aRLz*aLLy + aLLy*aRLx*aRLz*aRAy - aLLy*aRLx*aRLz*aRLy + aLLy*aRLx*aRLz*aLLy - aLLy*aRLy*aRLz*aRAx + aLLy*aRLy*aRLz*aRLx - aLLy*aRLy*aRLz*aLLx),
    f_LA: (aLLx*aRAy*aRLy*dRAz*gz - aLLx*aRAy*aRLy*dRLz*gz - aLLx*aRAy*aRLz*dRAy*gz + aLLx*aRAy*aRLz*dRLy*gz + aLLx*aRAz*aRLy*dRAy*gz - aLLx*aRAz*aRLy*dRLy*gz - aLLx*aRAz*aRLz*dRAy*gy + aLLx*aRAz*aRLz*dRLy*gy - aLLx*aRLy*aRLz*dRAy*gy + aLLx*aRLy*aRLz*dRLy*gy - aLLy*aRAx*aRLy*dRAz*gz + aLLy*aRAx*aRLy*dRLz*gz + aLLy*aRAx*aRLz*dRAy*gz - aLLy*aRAx*aRLz*dRLy*gz - aLLy*aRAz*aRLx*dRAy*gz + aLLy*aRAz*aRLx*dRLy*gz + aLLy*aRAz*aRLz*dRAx*gy - aLLy*aRAz*aRLz*dRLx*gy + aLLy*aRLx*aRLz*dRAy*gy - aLLy*aRLx*aRLz*dRLy*gy + aLLz*aRAx*aRLy*dRAy*gz - aLLz*aRAx*aRLy*dRLy*gz - aLLz*aRAx*aRLz*dRAy*gy + aLLz*aRAx*aRLz*dRLy*gy + aLLz*aRAy*aRLx*dRAy*gz - aLLz*aRAy*aRLx*dRLy*gz - aLLz*aRAy*aRLz*dRAx*gy + aLLz*aRAy*aRLz*dRLx*gy - aLLz*aRLx*aRLy*dRAy*gy + aLLz*aRLx*aRLy*dRLy*gy + aRAx*aRLy*aRLz*dRAy*gy - aRAx*aRLy*aRLz*dRLy*gy - aRAy*aRLx*aRLz*dRAy*gy + aRAy*aRLx*aRLz*dRLy*gy + aRAy*aRLy*aRLz*dRAx*gy - aRAy*aRLy*aRLz*dRLx*gy - aLAx*aRAy*aRLy*dRAz*gz + aLAx*aRAy*aRLy*dRLz*gz + aLAx*aRAy*aRLz*dRAy*gz - aLAx*aRAy*aRLz*dRLy*gz - aLAx*aRAz*aRLy*dRAy*gz + aLAx*aRAz*aRLy*dRLy*gz + aLAx*aRAz*aRLz*dRAy*gy - aLAx*aRAz*aRLz*dRLy*gy + aLAx*aRLy*aRLz*dRAy*gy - aLAx*aRLy*aRLz*dRLy*gy + aLAy*aRAx*aRLy*dRAy*gz - aLAy*aRAx*aRLy*dRLy*gz - aLAy*aRAx*aRLz*dRAy*gy + aLAy*aRAx*aRLz*dRLy*gy + aLAy*aRAy*aRLx*dRAy*gz - aLAy*aRAy*aRLx*dRLy*gz - aLAy*aRLx*aRLy*dRAy*gy + aLAy*aRLx*aRLy*dRLy*gy + aLAz*aRAx*aRLy*dRAy*gy - aLAz*aRAx*aRLy*dRLy*gy - aLAz*aRAy*aRLx*dRAy*gy + aLAz*aRAy*aRLx*dRLy*gy + aLAz*aRLx*aRLy*dRAy*gy - aLAz*aRLx*aRLy*dRLy*gy - aRAx*aRLy*aRLz*dLAy*gy + aRAx*aRLy*aRLz*dRLy*gy + aRAy*aRLx*aRLz*dLAy*gy - aRAy*aRLx*aRLz*dRLy*gy - aRAy*aRLy*aRLz*dLAx*gy + aRAy*aRLy*aRLz*dRLx*gy)/(aLAx*aLLy*aRLy*aRAz - aLAx*aLLy*aRLy*aRLz + aLAx*aLLy*aRLy*aLLz - aLAx*aLLy*aRLz*aRAy + aLAx*aLLy*aRLz*aRLy - aLAx*aLLy*aRLz*aLLy - aLAx*aLLz*aRLy*aRAy + aLAx*aLLz*aRLy*aRLy - aLAx*aLLz*aRLy*aLLy + aLAx*aLLz*aRLz*aRAy - aLAx*aLLz*aRLz*aRLy + aLAx*aLLz*aRLz*aLLy + aLAx*aRLy*aRLz*aRAy - aLAx*aRLy*aRLz*aRLy + aLAx*aRLy*aRLz*aLLy - aLAy*aLLx*aRLy*aRAz + aLAy*aLLx*aRLy*aRLz - aLAy*aLLx*aRLy*aLLz + aLAy*aLLx*aRLz*aRAx - aLAy*aLLx*aRLz*aRLx + aLAy*aLLx*aRLz*aLLx + aLAy*aLLz*aRLx*aRAy - aLAy*aLLz*aRLx*aRLy + aLAy*aLLz*aRLx*aLLy - aLAy*aLLz*aRLz*aRAx + aLAy*aLLz*aRLz*aRLx - aLAy*aLLz*aRLz*aLLx - aLAy*aRLx*aRLz*aRAx + aLAy*aRLx*aRLz*aRLx - aLAy*aRLx*aRLz*aLLx + aLAz*aLLx*aRLy*aRAy - aLAz*aLLx*aRLy*aRLy + aLAz*aLLx*aRLy*aLLy - aLAz*aLLx*aRLz*aRAy + aLAz*aLLx*aRLz*aRLy - aLAz*aLLx*aRLz*aLLy - aLAz*aLLy*aRLx*aRAy + aLAz*aLLy*aRLx*aRLy - aLAz*aLLy*aRLx*aLLy + aLAz*aLLy*aRLz*aRAx - aLAz*aLLy*aRLz*aRLx + aLAz*aLLy*aRLz*aLLx + aLAz*aRLx*aRLy*aRAx - aLAz*aRLx*aRLy*aRLx + aLAz*aRLx*aRLy*aLLx - aLLx*aRLy*aRLz*aRAy + aLLx*aRLy*aRLz*aRLy - aLLx*aRLy*aRLz*aLLy + aLLy*aRLx*aRLz*aRAy - aLLy*aRLx*aRLz*aRLy + aLLy*aRLx*aRLz*aLLy - aLLy*aRLy*aRLz*aRAx + aLLy*aRLy*aRLz*aRLx - aLLy*aRLy*aRLz*aLLx),
    # ... (f_RL and f_LL substitution expressions follow here in your original file)
})

# Substitute geometric expressions into F_sub to get F as a function of marker positions etc.
F = F_sub.subs({
    aRAx: RHx - CMx,
    aRAy: RHy - CMy,
    aRAz: RHz - CMz,
    aLAx: LHx - CMx,
    aLAy: LHy - CMy,
    aLAz: LHz - CMz,
    aRLx: CMx - RFx,
    aRLy: CMy - RFy,
    aRLz: CMz - RFz,
    aLLx: CMx - LFx,
    aLLy: CMy - LFy,
    aLLz: CMz - LFz,
    dRAx: RHx - CMx,  # etc. (your file already had full d* substitutions and gravity torque terms)
    dRAy: RHy - CMy,
    dRAz: RHz - CMz,
    dLAx: LHx - CMx,
    dLAy: LHy - CMy,
    dLAz: LHz - CMz,
    dRLx: RFx - CMx,
    dRLy: RFy - CMy,
    dRLz: RFz - CMz,
    dLLx: LFx - CMx,
    dLLy: LFy - CMy,
    dLLz: LFz - CMz,
    dMMx: CMx - CMx,  # placeholder if needed
    dMMy: CMy - CMy,
    dMMz: CMz - CMz,
    gx: 0,
    gy: -weight,
    gz: 0,
})

# Partial derivatives of F w.r.t. center of mass coordinates
df_dx = sp.diff(F, CMx)
df_dy = sp.diff(F, CMy)
df_dz = sp.diff(F, CMz)


def solve_contact_forces_cvxpy(marker_positions, total_mass, gravity, nonnegativity=True):
    """
    Solve for limb contact force magnitudes [f_LA, f_RA, f_RL, f_LL]
    by minimizing hand effort (f_LA + f_RA) subject to static equilibrium.

    marker_positions: dict with keys
        "RFAradius", "LFAradius", "RTOE", "LTOE", "MASSCENT", "MOMENTCENT"
        each -> [x,y,z] in meters (OpenSim ground frame)
    total_mass: subject mass in kg
    gravity: gravitational acceleration magnitude (>0, m/s^2)
    nonnegativity: if True, constrain all forces >= 0

    Returns dict:
        {
            "f_LA": ...,
            "f_RA": ...,
            "f_RL": ...,
            "f_LL": ...,
            "status": optimization_status,
            "objective_value": objective_value
        }
    """
    import numpy as _np

    # weight vector (N)
    weight_local = total_mass * gravity
    gx_local, gy_local, gz_local = 0.0, -weight_local, 0.0

    # extract points
    RH = _np.array(marker_positions["RFAradius"], dtype=float)   # right hand
    LH = _np.array(marker_positions["LFAradius"], dtype=float)   # left hand
    RF = _np.array(marker_positions["RTOE"], dtype=float)        # right toe
    LF = _np.array(marker_positions["LTOE"], dtype=float)        # left toe
    CM = _np.array(marker_positions["MASSCENT"], dtype=float)    # COM
    MM = _np.array(marker_positions["MOMENTCENT"], dtype=float)  # moment ref

    # direction vectors for each contact force
    # arms pull from COM toward handhold, legs push from foot toward COM
    aRA = RH - CM  # right arm
    aLA = LH - CM  # left  arm
    aRL = CM - RF  # right leg
    aLL = CM - LF  # left  leg

    # moment arms (contact point relative to moment center)
    dRA = RH - MM
    dLA = LH - MM
    dRL = RF - MM
    dLL = LF - MM

    # COM offset from moment center (for gravity torque)
    dMM = CM - MM

    # Build linear system A @ F = b
    # Unknown order: [f_LA, f_RA, f_RL, f_LL]
    row1 = [aLA[0], aRA[0], aRL[0], aLL[0]]  # force balance x
    row2 = [aLA[1], aRA[1], aRL[1], aLL[1]]  # force balance y
    row3 = [aLA[2], aRA[2], aRL[2], aLL[2]]  # force balance z

    # Moment balance around MM:
    # f_RA*(dRAy*aRAz - dRAz*aRAy) + ... + (dMMy*gz - dMMz*gy) = 0
    coef_LA = (dLA[1] * aLA[2] - dLA[2] * aLA[1])
    coef_RA = (dRA[1] * aRA[2] - dRA[2] * aRA[1])
    coef_RL = (dRL[1] * aRL[2] - dRL[2] * aRL[1])
    coef_LL = (dLL[1] * aLL[2] - dLL[2] * aLL[1])
    row4 = [coef_LA, coef_RA, coef_RL, coef_LL]

    A = _np.array([row1, row2, row3, row4], dtype=float)

    rhs1 = -gx_local
    rhs2 = -gy_local
    rhs3 = -gz_local
    rhs4 = -(dMM[1] * gz_local - dMM[2] * gy_local)
    b = _np.array([rhs1, rhs2, rhs3, rhs4], dtype=float)

    # Optimization variable
    Fvar = cp.Variable(4)

    objective = cp.Minimize(cp.sum_squares(Fvar))
    constraints = [A @ Fvar == b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)


    if Fvar.value is None:
        raise RuntimeError("Infeasible force optimization: no solution satisfies constraints.")

    return {
        "f_LA": float(Fvar.value[0]),
        "f_RA": float(Fvar.value[1]),
        "f_RL": float(Fvar.value[2]),
        "f_LL": float(Fvar.value[3]),
        "status": prob.status,
        "objective_value": float(prob.value),
    }


def GetFingerForces(CenterMass, der):
    model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
    model = osim.Model(model_path)
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        coord.setDefaultLocked(False)

    # Marker names and corresponding body parts
    marker_defs = {
        "RFAradius": hold_positions["hand_r"],  # must match names in model
        "LFAradius": hold_positions["hand_l"],
        "RTOE": hold_positions["toes_r"],
        "LTOE": hold_positions["toes_l"],
        "RFAtip": hold_positions["hand_r_tip"],  
        "LFAtip": hold_positions["hand_l_tip"],
        "MASSCENT": CenterMass
    }

    os.makedirs(output_dir, exist_ok=True)

    # Prepare files for Inverse Kinematics
    trc_file = output_dir + pose_name + "_markers.trc"
    # Time settings
    num_frames = 3
    start_time = 0.0
    end_time = 0.02
    time_array = np.linspace(start_time, end_time, num_frames)

    # Marker names and positions (in mm)
    marker_names = list(marker_defs.keys())

    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{trc_file}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"100.00\t100.00\t{num_frames}\t{len(marker_names)}\tmm\t100.00\t0\t{num_frames}",
        "Frame#\tTime\t" + "\t\t".join(marker_names),
        "\t\t" + "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(marker_names)))
    ]

    rows = []
    for i, t in enumerate(time_array, 1):
        row = [f"{i}", f"{t:.5f}"]
        for name in marker_names:
            x, y, z = [1000 * v for v in marker_defs[name]]  # Convert to mm
            row += [f"{x:.1f}", f"{y:.1f}", f"{z:.1f}"]
        rows.append("\t".join(row))

    with open(trc_file, "w") as f:
        f.write("\n".join(header + rows))


    # Inverse Kinematics
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setName(pose_name + "_IK")
    ik_tool.setMarkerDataFileName(trc_file)
    ik_tool.setStartTime(start_time)
    ik_tool.setEndTime(end_time)
    ik_tool.setOutputMotionFileName(output_dir + pose_name + "_pose.mot")
    ik_tool.set_results_directory(output_dir)
    ik_tool.run()




    model.finalizeConnections()
    state = model.initSystem()
    model.printToXML(output_dir + "model_with_contacts.osim")
    model = osim.Model(output_dir + "model_with_contacts.osim")
    model_path = output_dir + "model_with_contacts.osim"
    motion_file = os.path.join(output_dir, pose_name + "_pose.mot")

    state = model.initSystem()
    table = osim.TimeSeriesTable(motion_file)
    labels = table.getColumnLabels()
    row = table.getRowAtIndex(0)

    for i, label in enumerate(labels):
        if model.getCoordinateSet().contains(label):
            coord = model.getCoordinateSet().get(label)
            value_rad = np.deg2rad(row[i])
            coord.setValue(state, value_rad, True)

    model.realizePosition(state)

    # Calculate Finger Force
    marker_names = [
        "RFAradius", "LFAradius", "RTOE", "LTOE",
        "RShoulder", "LShoulder", "RHip", "LHip", "MASSCENT", "MOMENTCENT"
    ]

    marker_positions = {}
    for name in marker_names:
        if name in ["MASSCENT", "MOMENTCENT"]:
            body = model.getBodySet().get("torso")
            location = body.getPositionInGround(state)
        else:
            if name == "RFAradius":
                body = model.getBodySet().get("hand_r")
            elif name == "LFAradius":
                body = model.getBodySet().get("hand_l")
            elif name == "RTOE":
                body = model.getBodySet().get("toes_r")
            elif name == "LTOE":
                body = model.getBodySet().get("toes_l")
            else:
                body = model.getBodySet().get("torso")
            location = body.getPositionInGround(state)

        marker_positions[name] = [location.get(i) for i in range(3)]

    # Variable substitutions for symbolic F and dF/dx...
    Values = {
        RHx: marker_positions["RFAradius"][0],
        RHy: marker_positions["RFAradius"][1],
        RHz: marker_positions["RFAradius"][2],
        LHx: marker_positions["LFAradius"][0],
        LHy: marker_positions["LFAradius"][1],
        LHz: marker_positions["LFAradius"][2],
        RFx: marker_positions["RTOE"][0],
        RFy: marker_positions["RTOE"][1],
        RFz: marker_positions["RTOE"][2],
        LFx: marker_positions["LTOE"][0],
        LFy: marker_positions["LTOE"][1],
        LFz: marker_positions["LTOE"][2],
        CMx: marker_positions["MASSCENT"][0],
        CMy: marker_positions["MASSCENT"][1],
        CMz: marker_positions["MASSCENT"][2],
        gx: 0,
        gy: weight,
        gz: 0,
        dRAx: marker_positions["RFAradius"][0] - marker_positions["MOMENTCENT"][0],
        dRAy: marker_positions["RFAradius"][1] - marker_positions["MOMENTCENT"][1],
        dRAz: marker_positions["RFAradius"][2] - marker_positions["MOMENTCENT"][2],
        dLAx: marker_positions["LFAradius"][0] - marker_positions["MOMENTCENT"][0],
        dLAy: marker_positions["LFAradius"][1] - marker_positions["MOMENTCENT"][1],
        dLAz: marker_positions["LFAradius"][2] - marker_positions["MOMENTCENT"][2],
        dRLx: marker_positions["RTOE"][0] - marker_positions["MOMENTCENT"][0],
        dRLy: marker_positions["RTOE"][1] - marker_positions["MOMENTCENT"][1],
        dRLz: marker_positions["RTOE"][2] - marker_positions["MOMENTCENT"][2],
        dLLx: marker_positions["LTOE"][0] - marker_positions["MOMENTCENT"][0],
        dLLy: marker_positions["LTOE"][1] - marker_positions["MOMENTCENT"][1],
        dLLz: marker_positions["LTOE"][2] - marker_positions["MOMENTCENT"][2],
        dMMx: marker_positions["MASSCENT"][0] - marker_positions["MOMENTCENT"][0],
        dMMy: marker_positions["MASSCENT"][1] - marker_positions["MOMENTCENT"][1],
        dMMz: marker_positions["MASSCENT"][2] - marker_positions["MOMENTCENT"][2],
    }

    # Add frame to motion for visualization etc. (unchanged logic below)
    source_motion_file = motion_file
    summary_file = os.path.join(output_dir, "summary_pose.mot")

    table = osim.TimeSeriesTable(source_motion_file)
    first_row = table.getRowAtIndex(0)
    labels = table.getColumnLabels()

    if not os.path.exists(summary_file):
        with open(source_motion_file, "r") as fin:
            lines = fin.readlines()
        header_end_index = next(i for i, line in enumerate(lines) if "endheader" in line.lower()) + 1
        header_lines = lines[:header_end_index]

        with open(summary_file, "w") as fout:
            fout.writelines(header_lines)

        summary_table = osim.TimeSeriesTable()
        summary_table.setColumnLabels(labels)
        summary_table.appendRow(0.0, first_row)
        temp_path = summary_file + ".tmp"
        osim.STOFileAdapter.write(summary_table, temp_path)

        with open(temp_path, "r") as ftemp:
            temp_lines = ftemp.readlines()
        data_start = next(i for i, line in enumerate(temp_lines) if "endheader" in line.lower()) + 1
        with open(summary_file, "a") as fout:
            fout.writelines(temp_lines[data_start:])
        os.remove(temp_path)
        print("Created new summary file with original header.")
    else:
        # Load existing summary and append first row
        summary_table = osim.TimeSeriesTable(summary_file)
        last_time = summary_table.getIndependentColumn()[-1]
        summary_table.appendRow(last_time + 0.01, first_row)
        osim.STOFileAdapter.write(summary_table, summary_file)
        print("Appended first frame to existing summary.")

    forces = solve_contact_forces_cvxpy(marker_positions, total_mass, gravity)

    Values.update({
        f_LA: forces["f_LA"],
        f_RA: forces["f_RA"],
        f_RL: forces["f_RL"],
        f_LL: forces["f_LL"],
    })

    if der:
        return (
            float(df_dx.subs(Values)),
            float(df_dy.subs(Values)),
            float(df_dz.subs(Values)),
            forces  # optional: return the actual force split too
        )
    else:
        return float(F.subs(Values)), forces


def Visualize():
    print("Playing motion in OpenSim Visualizer...")

    
    #Optional: Add geometry path for visualization
    osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path)


    #Disable all muscles
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.set_appliesForce(False)

    #Finalize and load motion
    model.finalizeConnections()
    motion = osim.TimeSeriesTable(os.path.join(output_dir, "summary_pose.mot"))

    #Visualize the model with motion
    try:
        osim.VisualizerUtilities.showMotion(model, motion)
    except RuntimeError as e:
        if "VisualizerProtocol" in str(e):
            print("Visualizer closed by user. No problem.")
        else:
            raise  # Re-raise if it's a real problem

    #Reenable all muscles
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.set_appliesForce(True)

#GetFingerForces([0.3,1,0.5],True)
#GetFingerForces([1.0,0,0.5],True)
#Visualize()