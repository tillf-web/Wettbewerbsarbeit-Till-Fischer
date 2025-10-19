import opensim as osim

# === Paths ===
model_path = "C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/FullBody_SpineBase_WithLegsArmsMarkers_REALIGNED.osim"
motion_path = "C:/Users/till/Documents/OpenSim/4.5/Code/Python/ClimbingOptimization/output/static_climb_pose.mot"
geometry_path = "C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/Geometry"

# === Optional: Add geometry path for visualization
osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path)

# === Load model
model = osim.Model(model_path)
state = model.initSystem()

# Unlock everything
for i in range(model.getCoordinateSet().getSize()):
    model.getCoordinateSet().get(i).setLocked(state, False)

# === Disable all muscles
muscles = model.getMuscles()
for i in range(muscles.getSize()):
    muscle = muscles.get(i)
    muscle.set_appliesForce(False)


# === Finalize and load motion
model.finalizeConnections()
motion = osim.TimeSeriesTable(motion_path)

# === Visualize the model with motion


# Get the visualizer
vis = model.getVisualizer().updSimbodyVisualizer()

# Example: draw a red force arrow
origin = osim.Vec3(0, 1, 0)          # start point in ground
force  = osim.Vec3(0, 100, 0)        # vector direction

arrow = osim.simbody.DecorativeArrow(origin, origin + 0.01*force)
arrow.setColor(osim.Vec3(1, 0, 0))
arrow.setLineThickness(3.0)

# Attach arrow to ground
vis.addDecoration(model.getGround().getMobilizedBodyIndex(),
                  osim.simbody.Transform(),
                  arrow)

osim.VisualizerUtilities.showMotion(model, motion)