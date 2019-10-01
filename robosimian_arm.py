import pybullet as p
import time
import pybullet_data
import os
import sys

cid = p.connect(p.SHARED_MEMORY)

if (cid<0):
        p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0,0,-10)
useRealTimeSim = 1

p.setRealTimeSimulation(useRealTimeSim) # either this
plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))

p.resetSimulation()
print(pybullet_data.getDataPath())
rs = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"romans_urdf_files/send_to_sean/robosim_UCSB_no_wheel.urdf"), [0,0,1],flags=8, useFixedBase=1)
#rs = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"robosim_UCSB.urdf", [2,0,1],flags=8, useFixedBase=1))


while(True):
        dummy_var=1

