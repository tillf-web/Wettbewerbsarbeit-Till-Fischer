import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd



# === CONFIGURATION ===
model_path = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim"
output_dir = "C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/output2/"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Geometry"
pose_name = "static_climb"
total_mass = 75  # kg
gravity = 9.81
weight = total_mass * gravity
force_each = weight / 4
differneceFactor = 0.5


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
        #"MASSCENT": CenterMass
    }

    os.makedirs(output_dir, exist_ok=True)

    # === STEP 2: Create Marker Trajectory File (.trc) ===
    trc_file = output_dir + pose_name + "_markers.trc"
    # Time settings
    num_frames = 10
    start_time = 0.0
    end_time = 0.09
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

    # === STEP 7: Static Optimization ===
    # Unlock all coordinates
    coords = model.updCoordinateSet()
    for i in range(coords.getSize()):
        coord = coords.get(i)
        coord.setDefaultLocked(False)   # unlock for default state
        coord.setLocked(state, False)   # unlock in the initialized State

    # Save modified model (optional)
    model.printToXML(output_dir + "model_with_contacts.osim")


    setup_file = os.path.join(output_dir, pose_name + "_StaticOptimizationSetup.xml")
    motion_file = os.path.join(output_dir, pose_name + "_pose.mot")
    root = ET.Element("AnalyzeTool", name="static_climb_StaticOptimization")
    start_time=0
    end_time=0
    ET.SubElement(root, "model_file").text = model_path
    ET.SubElement(root, "replace_force_set").text = "false"
    ET.SubElement(root, "results_directory").text = output_dir
    ET.SubElement(root, "output_precision").text = "20"
    ET.SubElement(root, "initial_time").text = str(start_time)
    ET.SubElement(root, "final_time").text = str(end_time)

    # Add AnalysisSet with StaticOptimization
    analysis_set = ET.SubElement(root, "AnalysisSet", name="Analyses")
    objects = ET.SubElement(analysis_set, "objects")
    so = ET.SubElement(objects, "StaticOptimization", name="StaticOptimization")

    ET.SubElement(so, "on").text = "true"
    ET.SubElement(so, "use_model_force_set").text = "true"
    ET.SubElement(so, "activation_exponent").text = "2"
    ET.SubElement(so, "use_muscle_physiology").text = "true"
    ET.SubElement(so, "optimizer_convergence_criterion").text = "0.0001"
    ET.SubElement(so, "optimizer_max_iterations").text = "100"

    # Coordinates file and external loads
    ET.SubElement(root, "coordinates_file").text = motion_file
    ET.SubElement(root, "lowpass_cutoff_frequency_for_coordinates").text = "6"
    #ET.SubElement(root, "external_loads_file").text = external_loads_file

    # Add ForceReporter
    force_reporter = ET.SubElement(objects, "ForceReporter", name="ForceReporter")
    ET.SubElement(force_reporter, "on").text = "true"
    ET.SubElement(force_reporter, "steps_per_report").text = "1"
    ET.SubElement(force_reporter, "force_storage_file").text = "forces_reporter_output.sto"

    # === Save to XML ===
    tree = ET.ElementTree(root)
    os.makedirs(output_dir, exist_ok=True)
    tree.write(setup_file, encoding="UTF-8", xml_declaration=True)
    print(f"Static Optimization setup file written to: {setup_file}")

    # === STEP 3: Run Static Optimization from XML ===
    tool = osim.AnalyzeTool(setup_file)
    tool.run()

    print("Static Optimization completed. Results saved in:", output_dir)


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



    

    #get variables
    # Path to your Static Optimization .sto force file
    sto_file = "output/static_climb_StaticOptimization_StaticOptimization_force.sto"

    # Locate the data start (after 'endheader')
    with open(sto_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            header_row = i
            break

    # Read data into DataFrame
    df = pd.read_csv(sto_file, sep='\s+', skiprows=header_row + 1)

    # Actuator names — make sure these match what you named them
    actuators = ["hand_r_contact", "hand_l_contact", "toes_r_contact", "toes_l_contact"]

    # Filter first row at time = 0.0
    first_frame = df[df["time"] == 0.0]

    # Extract forces into variables
    hand_r_force = first_frame["hand_r_contact"].values[0] if "hand_r_contact" in df.columns else None
    hand_l_force = first_frame["hand_l_contact"].values[0] if "hand_l_contact" in df.columns else None
    toes_r_force = first_frame["toes_r_contact"].values[0] if "toes_r_contact" in df.columns else None
    toes_l_force = first_frame["toes_l_contact"].values[0] if "toes_l_contact" in df.columns else None

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

GetFingerForces([0,0,0])
GetFingerForces([0,0,0])
Visualize()