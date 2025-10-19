import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
import sympy as sp
import math


# === CONFIGURATION ===
model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
output_dir = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/output2/"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Geometry"
pose_name = "static_climb"
total_mass = 75  # kg
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



def GetFingerForces(CenterMass):
    model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
    model = osim.Model(model_path)
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

    # === STEP 2: Create Marker Trajectory File (.trc) ===
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


    # === STEP 3: Inverse Kinematics ===
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setName(pose_name + "_IK")
    ik_tool.setMarkerDataFileName(trc_file)
    ik_tool.setStartTime(start_time)
    ik_tool.setEndTime(end_time)
    ik_tool.setOutputMotionFileName(output_dir + pose_name + "_pose.mot")
    ik_tool.set_results_directory(output_dir)
    ik_tool.run()

    state = model.initSystem()
    constraint_defs = []

    #for label, world_pos in hold_positions.items():
    #    body_name = body_map[label]
    #    body_frame = osim.PhysicalFrame.safeDownCast(model.getBodySet().get(body_name.split("/")[0]))
    #    
    #    world_vec = osim.Vec3(*world_pos)
    #    local_vec = model.getGround().findStationLocationInAnotherFrame(state, world_vec, body_frame)
    #
    #    constraint_defs.append((label, body_frame, world_vec, local_vec))

    #for label, body_frame, world_vec, local_vec in constraint_defs:
    #    constraint = osim.PointConstraint()
    #    constraint.setName(f"{label}_constraint")
    #    constraint.setBody1ByName("/bodyset/"+body_frame.getName())
    #    constraint.setBody2ByName("/ground")
    #    constraint.setBody1PointLocation(local_vec)
    #    constraint.setBody2PointLocation(world_vec)

#        model.addConstraint(constraint)
 #       print(f"Added PointConstraint: {label} to {body_frame.getName()}")


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
    

    # Get desired marker names
    marker_names = [
        "RFAradius", "LFAradius", "RTOE", "LTOE", 
        "RShoulder", "LShoulder", "RHip", "LHip", "MASSCENT"
    ]

    # Prepare output
    marker_positions = {}

    for marker in model.getMarkerSet():
        name = marker.getName()
        if name in marker_names:
            pos = marker.getLocationInGround(state)
            marker_positions[name] = [pos.get(i) for i in range(3)]

    print("Marker positions at time 0.0 (in meters):")
    for name, pos in marker_positions.items():
        print(f"{name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    ForceComponents = {
        "RAx":marker_positions["RFAradius"][0]-marker_positions["RShoulder"][0],
        "RAy":marker_positions["RFAradius"][1]-marker_positions["RShoulder"][1],
        "RAz":marker_positions["RFAradius"][2]-marker_positions["RShoulder"][2],
        "LAx":marker_positions["LFAradius"][0]-marker_positions["LShoulder"][0],
        "LAy":marker_positions["LFAradius"][1]-marker_positions["LShoulder"][1],
        "LAz":marker_positions["LFAradius"][2]-marker_positions["LShoulder"][2],
        "RLx":marker_positions["RTOE"][0]-marker_positions["RHip"][0],
        "RLy":marker_positions["RTOE"][1]-marker_positions["RHip"][1],
        "RLz":marker_positions["RTOE"][2]-marker_positions["RHip"][2],
        "LLx":marker_positions["LTOE"][0]-marker_positions["LHip"][0],
        "LLy":marker_positions["LTOE"][1]-marker_positions["LHip"][1],
        "LLz":marker_positions["LTOE"][2]-marker_positions["LHip"][2],
    }
    # Example: substitute numeric values
    values = {
        F_a:ForceComponents["RAy"]/ForceComponents["RAx"], F_c:ForceComponents["RLy"]/ForceComponents["RLx"], F_r:ForceComponents["LLy"]/ForceComponents["LLx"], F_t:ForceComponents["LAy"]/ForceComponents["LAx"],
        F_b:ForceComponents["RAz"]/ForceComponents["RAx"], F_d:ForceComponents["RLz"]/ForceComponents["RLx"], F_s:ForceComponents["LLz"]/ForceComponents["LLx"], F_u:ForceComponents["LAz"]/ForceComponents["LAx"],
        F_dfy:marker_positions["RShoulder"][1]-marker_positions["MASSCENT"][1], F_dfz:marker_positions["RShoulder"][2]-marker_positions["MASSCENT"][2], F_dgy:marker_positions["RHip"][1]-marker_positions["MASSCENT"][1], F_dgz:marker_positions["RHip"][2]-marker_positions["MASSCENT"][2],
        F_dhy:marker_positions["LHip"][1]-marker_positions["MASSCENT"][1], F_dhz:marker_positions["LHip"][2]-marker_positions["MASSCENT"][2], F_djy:marker_positions["LShoulder"][1]-marker_positions["MASSCENT"][1], F_djz:marker_positions["LShoulder"][2]-marker_positions["MASSCENT"][2],
        F_p:weight
    }

    numeric_solution = {var: solution[var].subs(values).evalf() for var in solution}
    print(numeric_solution)


    #complete animation

    # === CONFIGURATION ===
    source_motion_file = motion_file
    summary_file = os.path.join(output_dir, "summary_pose.mot")

    # === Load the original .mot file to get the first frame
    table = osim.TimeSeriesTable(source_motion_file)
    first_row = table.getRowAtIndex(0)
    labels = table.getColumnLabels()

    if not os.path.exists(summary_file):
        # === Read and copy the full header from source .mot
        with open(source_motion_file, "r") as fin:
            lines = fin.readlines()
        header_end_index = next(i for i, line in enumerate(lines) if "endheader" in line.lower()) + 1
        header_lines = lines[:header_end_index]

        # === Write header to new file
        with open(summary_file, "w") as fout:
            fout.writelines(header_lines)

        # === Create and write one-row table to temp file
        summary_table = osim.TimeSeriesTable()
        summary_table.setColumnLabels(labels)
        summary_table.appendRow(0.0, first_row)
        temp_path = summary_file + ".tmp"
        osim.STOFileAdapter.write(summary_table, temp_path)

        # === Append data (excluding auto header) to final file
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


    hand_l_force=math.sqrt( pow(numeric_solution[F_x4],2)+pow(numeric_solution[F_x4]/values[F_t],2)+pow(numeric_solution[F_x4]/values[F_u],2))
    hand_r_force=math.sqrt( pow(numeric_solution[F_x1],2)+pow(numeric_solution[F_x1]/values[F_a],2)+pow(numeric_solution[F_x1]/values[F_b],2))
    print(hand_l_force)
    print(hand_r_force)
    return(hand_l_force+hand_r_force+(differneceFactor*abs(hand_l_force-hand_r_force)))



def Visualize():
    print("Playing motion in OpenSim Visualizer...")

    
    # === Optional: Add geometry path for visualization
    osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path)


    # === Disable all muscles
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.set_appliesForce(False)

    # === Finalize and load motion
    model.finalizeConnections()
    motion = osim.TimeSeriesTable(os.path.join(output_dir, "summary_pose.mot"))

    # === Visualize the model with motion
    try:
        osim.VisualizerUtilities.showMotion(model, motion)
    except RuntimeError as e:
        if "VisualizerProtocol" in str(e):
            print("Visualizer closed by user. No problem.")
        else:
            raise  # Re-raise if it's a real problem

    # === Reenable all muscles
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.set_appliesForce(True)

#GetFingerForces([0.3,1,0.5])
#GetFingerForces([1.0,0,0.5])
#Visualize()