import opensim as osim

# === Paths ===
model_path = "C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/FullBody_SpineBase_WithLegsArmsMarkers_REALIGNED.osim"
motion_path = "C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/static_climb_pose.mot"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/Geometry"

# === Optional: Add geometry path for visualization
osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path)

model = osim.Model("C:/Users/till/OneDrive - Kantonsschule Wettingen/Schule/MATURARBEIT/ClimbingOptimization/Rajagopal2015 wall.osim")
model.finalizeConnections()
model.setUseVisualizer(True)
state = model.initSystem()
model.getVisualizer().show(state)
model.finalizeConnections()

