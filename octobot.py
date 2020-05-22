import pybullet as p
import time
import pybullet_data #helps locate urdf file
import os
import sys
import numpy as np
import pdb



clientId = p.connect(p.GUI) #chose render mode

print(clientId)
if (clientId<0): #safety rendering
	clientId = p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0,0,-10)
useRealTimeSim = 1 #if set to 0, stepping function must be used to step simulation
p.setRealTimeSimulation(useRealTimeSim)

#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "plane.urdf" ))

plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/octopus_simple.urdf" ))

#blockPos = [0,0,3]
#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "cube_small.urdf" ) , basePosition = blockPos    )

#p.resetSimulation()
#robotId = p.loadURDF( os.path.join(pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/my_robot.urdf" ) )


angles = p.calculateInverseKinematics(bodyUniqueId=plane, endEffectorLinkIndex=0, targetPosition = [-1 , 0,0  ], solver=p.IK_DLS )

############uncomment this ##set target position
#p.setJointMotorControl2(bodyUniqueId=plane, jointIndex=0, controlMode=p.POSITION_CONTROL,targetPosition=0.5 )

#state=p.getJointState(bodyUniqueId=plane, jointIndex=0, physicsClientId=clientId)

#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.POSITION_CONTROL,targetPosition=0.5, 
##currentPosition=state[0] , 
#force=1) #move link to target position


#uncomment this #lockJoint
p.createConstraint(parentBodyUniqueId=plane, parentLinkIndex=-1, childBodyUniqueId=plane, childLinkIndex=0, jointType=p.JOINT_FIXED, jointAxis=[0,0,0], parentFramePosition=[2,2,2], childFramePosition=[2,2,2]  )

#########LOCK joint with force
#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=10000)#lock joint with large force

#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=1, force=0) #free all joint constraints with force 0


while(1):
	print(angles)
		
	state = p.getJointState(bodyUniqueId=plane, jointIndex = 0, physicsClientId= clientId) 
	print(state)
	print(state[0])
	print(angles)
	
	

#while(1):
#	print(444444444)
