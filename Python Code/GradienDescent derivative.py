from CalculateOpenSimModel_derivative import *

#Mittelpunkt finden
CentX=(hold_positions["hand_r"][0]+hold_positions["hand_l"][0]+hold_positions["toes_r"][0]+hold_positions["toes_l"][0])/4
CentY=(hold_positions["hand_r"][1]+hold_positions["hand_l"][1]+hold_positions["toes_r"][1]+hold_positions["toes_l"][1])/4
CentZ=(hold_positions["hand_r"][2]+hold_positions["hand_l"][2]+hold_positions["toes_r"][2]+hold_positions["toes_l"][2])/4
center_point = (CentX,CentY,CentZ)

#Generieren einer Startposition im Umkreis des Mittelpunktes
def random_point_in_sphere(center=(0, 0, 0), radius=1):
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    r = radius * np.cbrt(np.random.rand())
    point = np.array(center) + r * vec
    return tuple(point)

#Faktor für den Gradient Descent
alpha=0.00000000000000000001

#Durchlaufen des Gradient Descents (50 Durchläufe)
s=[]
for _ in range(50):
    P=random_point_in_sphere(center_point)
    x=P[0]
    y=P[1]
    z=P[2]
    #Ausschliesen von Startpositionen auf der anderen Seite der Kletterwand
    if(y>1.53*(1.4-x)+0.7 or x>1.4):
        continue
    lastJ=[0,0,0]
    #Einzlene Schritte des GRadient descents (20 Schritte maximal)
    for i in range(20):
        #Berechnung von Korperposition und Fingerkraft
        J=GetFingerForces([x,y,z],1)
        #Beenden bei Fehlerhaften Berechnung (Division durch null)
        if(len(list(J))<3):
            x+=alpha*list(lastJ)[0]
            y+=alpha*list(lastJ)[1]
            z+=alpha*list(lastJ)[2]
            break
        #Anpassen des Schwerpunktes
        x-=alpha*list(J)[0]
        y-=alpha*list(J)[1]
        z-=alpha*list(J)[2]
        #Beenden bei Schwerpunkt auf der anderen Seite der Kletterwand 
        if(y>1.53*(1.4-x)+0.7 or x>1.4):
            x+=alpha*list(J)[0]
            y+=alpha*list(J)[1]
            z+=alpha*list(J)[2]
            break
        lastJ=J
    V=GetFingerForces([x,y,z],0)
    GetFingerForces([100,100,100],0)
    #if(len(list(J))<3) or V==0:
    #    continue
    if(lastJ==[0,0,0]):
        continue
    s.append((V,[x,y,z]))

s.sort(reverse=True)
for i in s:
    print(i[0])
    GetFingerForces(i[1], 1)


Visualize()
