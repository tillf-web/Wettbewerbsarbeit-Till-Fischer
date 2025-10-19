from CalculateOpenSimModel_forces import *

x=0.7001
y=1.3001
z=0.5001
lastX=0.7
lastY=1.3
lastZ=0.5

J=GetFingerForces([lastX,lastY,lastZ])
newJ=GetFingerForces([x,y,z])
#Visualize()

alpha=0.0000001

while(abs(newJ-J)>10):
    
    if(x-lastX!=0):
        v=x-alpha*((newJ-J)/(x-lastX))
        lastX=x
        x=v
    if(y-lastY!=0):
        v=y-alpha*((newJ-J)/(y-lastY))
        lastY=y
        y=v
    if(z-lastZ!=0):
        v=z-alpha*((newJ-J)/(z-lastZ))
        lastZ=z
        z=v
    J=newJ
    newJ=GetFingerForces([x,y,z])

Visualize()
