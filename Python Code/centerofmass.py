import opensim as osim

# Load your model
model = osim.Model("C:/Users/till/Documents/OpenSim/4.5/Models/Spine_children_models_OS4_v2/FullBody_SpineBase_WithLegsArmsMarkers_REALIGNED.osim")
state = model.initSystem()

# Compute the center of mass in ground
com = model.getMatterSubsystem().calcSystemMassCenterLocationInGround(state)

# Print center of mass coordinates
print("Center of Mass Coordinates:")
print(f"X: {com.get(0):.3f} m")
print(f"Y: {com.get(1):.3f} m")
print(f"Z: {com.get(2):.3f} m")
