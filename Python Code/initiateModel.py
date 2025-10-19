import opensim as osim
import numpy as np
import os
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
model_path = "C:/Users/till/Documents/OpenSim/4.5/Models/FullBodyModel-4.0/Rajagopal2015.osim"
output_dir = "C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/"
pose_name = "static_climb"
total_mass = 75  # kg
gravity = 9.81
weight = total_mass * gravity
force_each = weight / 4

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


os.makedirs(output_dir, exist_ok=True)

# === STEP 1: Load and Prepare Model ===
model = osim.Model(model_path)
state = model.initSystem()


#marker_set = osim.MarkerSet()
#model.set_MarkerSet(marker_set)  # Attach empty MarkerSet to the model

# Add virtual markers at hold positions
#for name, (body_name, location) in marker_defs.items():
#    marker = osim.Marker()
#    marker.setName(name)
#    marker.set_location(osim.Vec3(*location))
#    marker.connectSocket_parent_frame(model.getBodySet().get(body_name))
#    marker_set.cloneAndAppend(marker)



model.finalizeConnections()

#load model markrs
model_markers = [model.getMarkerSet().get(i).getName() for i in range(model.getMarkerSet().getSize())]

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

# === STEP 4: Generate External Forces ===
sto_file = output_dir + pose_name + "_forces.sto"
contact_bodies = list(hold_positions.keys())

# Headers for force and point of application
force_point_headers = []
for body in contact_bodies:
    force_point_headers += [
        f"{body}_fx", f"{body}_fy", f"{body}_fz",
        f"{body}_px", f"{body}_py", f"{body}_pz"
    ]

# Create force data for each time point
force_rows = []
for t in time_array:
    row = [f"{t:.5f}"]
    for body in contact_bodies:
        # Constant vertical force and constant application point
        row += [
            "0.0", f"{force_each:.2f}", "0.0",  # fx, fy, fz
            f"{hold_positions[body][0]:.5f}",
            f"{hold_positions[body][1]:.5f}",
            f"{hold_positions[body][2]:.5f}",
        ]
    force_rows.append("\t".join(row))

# Write to .sto
with open(sto_file, "w") as f:
    f.write("External Forces\n")
    f.write("version=1\n")
    f.write(f"nRows={len(force_rows)}\n")
    f.write(f"nColumns={len(force_point_headers) + 1}\n")
    f.write("inDegrees=yes\n")
    f.write("endheader\n")
    f.write("time\t" + "\t".join(force_point_headers) + "\n")
    f.write("\n".join(force_rows) + "\n")


# === STEP 5: External Loads XML ===
external_xml = output_dir + pose_name + "_ExternalLoads.xml"
external_loads = osim.ExternalLoads()
external_loads.setDataFileName("C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/"+pose_name+"_forces.sto")

for body in contact_bodies:
    ext = osim.ExternalForce()
    ext.setName(f"{body}_hold")
    ext.set_applied_to_body(body)
    ext.set_force_expressed_in_body("ground")
    ext.set_point_expressed_in_body("ground")
    ext.set_data_source_name(sto_file)
    ext.set_force_identifier(f"{body}_f")   # force vector columns like foot_r_fy
    #ext.set_point_identifier(f"{body}_p") 

external_loads.printToXML(external_xml)

# === STEP 6: Inverse Dynamics ===
id_tool = osim.InverseDynamicsTool()
id_tool.setModel(model)
id_tool.setCoordinatesFileName(output_dir + pose_name + "_pose.mot")
id_tool.setExternalLoadsFileName(external_xml)
id_tool.setStartTime(start_time)
id_tool.setEndTime(end_time)
id_tool.setResultsDir(output_dir)
id_tool.setOutputGenForceFileName(pose_name + "_ID")
id_tool.setLowpassCutoffFrequency(6.0)
id_tool.run()

# === STEP 7: Static Optimization ===
setup_file = os.path.join(output_dir, pose_name + "_StaticOptimizationSetup.xml")
motion_file = os.path.join(output_dir, pose_name + "_pose.mot")
external_loads_file = os.path.join(output_dir, pose_name + "_ExternalLoads.xml")
output_file_name = pose_name + "_StaticOptimizationResults"

#so_tool = osim.StaticOptimization()
#so_tool.setName(pose_name + "_StaticOptimization")
#so_tool.setModel(model)
#so_tool.setStartTime(0.0)
#so_tool.setEndTime(0.01)
#so_tool.setUseModelForceSet(True)

# Save base XML
#so_tool.printToXML(setup_file)

root = ET.Element("AnalyzeTool", name="static_climb_StaticOptimization")

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
ET.SubElement(root, "external_loads_file").text = external_loads_file

# === Save to XML ===
tree = ET.ElementTree(root)
os.makedirs(output_dir, exist_ok=True)
tree.write(setup_file, encoding="UTF-8", xml_declaration=True)
print(f"Static Optimization setup file written to: {setup_file}")

# === STEP 3: Run Static Optimization from XML ===
tool = osim.AnalyzeTool(setup_file)
tool.run()

print("✅ Static Optimization completed. Results saved in:", output_dir)

