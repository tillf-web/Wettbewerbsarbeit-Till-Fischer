import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
import sympy as sp
import math


# Configuration
model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
output_dir = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/output2/"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Geometry"
pose_name = "static_climb"
total_mass = 65  # kg
gravity = 9.81
weight = total_mass * gravity
force_each = weight / 4
differneceFactor = 0


# === STEP 1: Load and Prepare Model ===
model = osim.Model(model_path)

# Optional: unlock defaults before initSystem
for i in range(model.getCoordinateSet().getSize()):
    coord = model.getCoordinateSet().get(i)
    coord.setDefaultLocked(False)

# Initialize
state = model.initSystem()

# Holds you want to extract
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

# Marker names and corresponding body parts
marker_defs = {
    "RFAradius": hold_positions["hand_r"],  # must match names in model
    "LFAradius": hold_positions["hand_l"],
    "RTOE": hold_positions["toes_r"],
    "LTOE": hold_positions["toes_l"],
}


body_map = {
    "hand_l": "hand_l/hand_l_geom_16",
    "hand_r": "hand_r/hand_r_geom_16",
    "toes_l": "toes_l/toes_l_geom_1",
    "toes_r": "toes_r/toes_r_geom_1",
}


if os.path.exists(os.path.join(output_dir, "summary_pose.mot")):
    os.remove(os.path.join(output_dir, "summary_pose.mot"))


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
    aRLx: CMx - RFx,
    aRLy: CMy - RFy,
    aRLz: CMz - RFz,
    aLLx: CMx - LFx,
    aLLy: CMy - LFy,
    aLLz: CMz - LFz,
})

# Partial derivatives
df_dx = sp.diff(F, CMx)
df_dy = sp.diff(F, CMy)
df_dz = sp.diff(F, CMz)



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
    # Load motion file (in degrees)
    table = osim.TimeSeriesTable(motion_file)
    labels = table.getColumnLabels()
    row = table.getRowAtIndex(0)

    # Apply the first frame of motion
    for i, label in enumerate(labels):
        if model.getCoordinateSet().contains(label):
            coord = model.getCoordinateSet().get(label)
            value_rad = np.deg2rad(row[i])  # convert from degrees to radians
            coord.setValue(state, value_rad, True)

    model.realizePosition(state)
    
    # Calculate Finger Force

    # Get desired marker names
    marker_names = [
        "RFAradius", "LFAradius", "RTOE", "LTOE", 
        "RShoulder", "LShoulder", "RHip", "LHip", "MASSCENT", "MOMENTCENT"
    ]

    # Read positions of markers
    marker_positions = {}

    for marker in model.getMarkerSet():
        name = marker.getName()
        if name in marker_names:
            pos = marker.getLocationInGround(state)
            marker_positions[name] = [pos.get(i) for i in range(3)]


    # Variable definitions
    Values = {
        RHx:marker_positions["RFAradius"][0],
        RHy:marker_positions["RFAradius"][1],
        RHz:marker_positions["RFAradius"][2],
        LHx:marker_positions["LFAradius"][0],
        LHy:marker_positions["LFAradius"][1],
        LHz:marker_positions["LFAradius"][2],
        RFx:marker_positions["RTOE"][0],
        RFy:marker_positions["RTOE"][1],
        RFz:marker_positions["RTOE"][2],
        LFx:marker_positions["LTOE"][0],
        LFy:marker_positions["LTOE"][1],
        LFz:marker_positions["LTOE"][2],
        CMx:marker_positions["MASSCENT"][0],
        CMy:marker_positions["MASSCENT"][1],
        CMz:marker_positions["MASSCENT"][2],
        gx:0,
        gy:-weight,
        gz:0,
        dRAx:marker_positions["RFAradius"][0]-marker_positions["MOMENTCENT"][0],
        dRAy:marker_positions["RFAradius"][1]-marker_positions["MOMENTCENT"][1],
        dRAz:marker_positions["RFAradius"][2]-marker_positions["MOMENTCENT"][2],
        dLAx:marker_positions["LFAradius"][0]-marker_positions["MOMENTCENT"][0],
        dLAy:marker_positions["LFAradius"][1]-marker_positions["MOMENTCENT"][1],
        dLAz:marker_positions["LFAradius"][2]-marker_positions["MOMENTCENT"][2],
        dRLx:marker_positions["RTOE"][0]-marker_positions["MOMENTCENT"][0],
        dRLy:marker_positions["RTOE"][1]-marker_positions["MOMENTCENT"][1],
        dRLz:marker_positions["RTOE"][2]-marker_positions["MOMENTCENT"][2],
        dLLx:marker_positions["LTOE"][0]-marker_positions["MOMENTCENT"][0],
        dLLy:marker_positions["LTOE"][1]-marker_positions["MOMENTCENT"][1],
        dLLz:marker_positions["LTOE"][2]-marker_positions["MOMENTCENT"][2],
        dMMx:marker_positions["MASSCENT"][0]-marker_positions["MOMENTCENT"][0],
        dMMy:marker_positions["MASSCENT"][1]-marker_positions["MOMENTCENT"][1],
        dMMz:marker_positions["MASSCENT"][2]-marker_positions["MOMENTCENT"][2],
    }
   
    #Add frame to motion for Visualization

    #configuration
    source_motion_file = motion_file
    summary_file = os.path.join(output_dir, "summary_pose.mot")

    #Load the original .mot file to get the first frame
    table = osim.TimeSeriesTable(source_motion_file)
    first_row = table.getRowAtIndex(0)
    labels = table.getColumnLabels()

    if not os.path.exists(summary_file):
        #Read and copy the full header from source .mot
        with open(source_motion_file, "r") as fin:
            lines = fin.readlines()
        header_end_index = next(i for i, line in enumerate(lines) if "endheader" in line.lower()) + 1
        header_lines = lines[:header_end_index]

        #Write header to new file
        with open(summary_file, "w") as fout:
            fout.writelines(header_lines)

        #Create and write one-row table to temp file
        summary_table = osim.TimeSeriesTable()
        summary_table.setColumnLabels(labels)
        summary_table.appendRow(0.0, first_row)
        temp_path = summary_file + ".tmp"
        osim.STOFileAdapter.write(summary_table, temp_path)

        #Append data (excluding auto header) to final file
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


    #return derrivative or funciton
    if der:
        return({df_dx.subs(Values),df_dy.subs(Values),df_dz.subs(Values)})
    else:
        return(F.subs(Values))



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

