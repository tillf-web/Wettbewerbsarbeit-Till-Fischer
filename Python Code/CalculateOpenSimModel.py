import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd



# === CONFIGURATION ===
model_path = "C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/FullBody_SpineBase_WithLegsArmsMarkers_REALIGNED.osim"#"C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Rajagopal2015.osim"
output_dir = "C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/Geometry"
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

# Unlock actual current state
for i in range(model.getCoordinateSet().getSize()):
    model.getCoordinateSet().get(i).setLocked(state, False)

# Positions in world frame (meters)
hold_positions = {
    "hand_l": [0.5,1.8,-0.1],
    "hand_r": [0.5, 2.0, 0.5],
    "toes_l": [1.2, 0.3, -0.1],
    "toes_r": [1.2, 0.7, 0.7],
}
# Marker names and corresponding body parts
marker_defs = {
    "RFAradius": hold_positions["hand_r"],  # must match names in model
    "LFAradius": hold_positions["hand_l"],
    "RTOE": hold_positions["toes_r"],
    "LTOE": hold_positions["toes_l"],
}
# STEP 3: Add PointActuators at Contact Locations ===
for label, position in hold_positions.items():
    actuator = osim.PointActuator()
    actuator.setName(f"{label}_contact")
    
    # Attach to the body associated with this marker
    # This assumes your model has markers named the same as labels (e.g., RTOE, LFAradius)
    body_name = None
    for i in range(model.getMarkerSet().getSize()):
        marker = model.getMarkerSet().get(i)
        if marker.getName() in marker_defs and marker_defs[marker.getName()] == position:
            body_name = marker.getParentFrame().getName()
            break
    if body_name is None:
        print(f"Warning: Could not find body for contact point '{label}' — skipping.")
        continue
    actuator.set_body(body_name)  
    actuator.set_point(osim.Vec3(*position))  # Global point of contact
    actuator.set_direction(osim.Vec3(0, 1, 0))  # +Y direction = normal force
    actuator.set_optimal_force(1000)  # Arbitrary large value, gets scaled
    actuator.set_force_is_global(True)
    model.addForce(actuator)
model.finalizeConnections()
model.printToXML(output_dir + "model_with_contacts.osim")
model = osim.Model(output_dir + "model_with_contacts.osim")
model_path = output_dir + "model_with_contacts.osim"

if os.path.exists(os.path.join(output_dir, "summary_pose.mot")):
    os.remove(os.path.join(output_dir, "summary_pose.mot"))

def GetFingerForces(CenterMass):
    
    # Marker names and corresponding body parts
    marker_defs = {
        "RFAradius": hold_positions["hand_r"],  # must match names in model
        "LFAradius": hold_positions["hand_l"],
        "RTOE": hold_positions["toes_r"],
        "LTOE": hold_positions["toes_l"],
        "MASSCENT": CenterMass
    }

    os.makedirs(output_dir, exist_ok=True)

    # === STEP 2: Create Marker Trajectory File (.trc) ===
    trc_file = output_dir + pose_name + "_markers.trc"
    # Time settings
    num_frames = 10
    start_time = 0.0
    end_time = 0.09
    time_array = np.linspace(start_time, end_time, num_frames)

    # Marker names and positions
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

    # === STEP 6: Inverse Dynamics ===
    id_tool = osim.InverseDynamicsTool()
    id_tool.setModel(model)
    id_tool.setCoordinatesFileName(output_dir + pose_name + "_pose.mot")
    #id_tool.setExternalLoadsFileName(external_xml)
    id_tool.setStartTime(start_time)
    id_tool.setEndTime(end_time)
    id_tool.setResultsDir(output_dir)
    id_tool.setOutputGenForceFileName(pose_name + "_ID")
    id_tool.setLowpassCutoffFrequency(6.0)
    id_tool.run()

    # === STEP 7: Static Optimization ===

    setup_file = os.path.join(output_dir, pose_name + "_StaticOptimizationSetup.xml")
    motion_file = os.path.join(output_dir, pose_name + "_pose.mot")
    root = ET.Element("AnalyzeTool", name="static_climb_StaticOptimization")
    start_time=0
    end_time=0.01
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

    # === Save to XML ===
    tree = ET.ElementTree(root)
    os.makedirs(output_dir, exist_ok=True)
    tree.write(setup_file, encoding="UTF-8", xml_declaration=True)
    print(f"Static Optimization setup file written to: {setup_file}")

    # === STEP 3: Run Static Optimization from XML ===
    tool = osim.AnalyzeTool(setup_file)
    #tool.run()

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

GetFingerForces([0,2,0.5])
GetFingerForces([0,2,0.5])
Visualize()