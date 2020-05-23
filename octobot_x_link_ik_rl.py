import pybullet as p
import time
import pybullet_data #helps locate urdf file
import os
import sys
import numpy as np
import pdb



clientId = p.connect(p.GUI) #choose render mode

print(clientId)
if (clientId<0): #safety rendering
	clientId = p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0,0,-10)
useRealTimeSim = 1 #if set to 0, stepping function must be used to step simulation
p.setRealTimeSimulation(useRealTimeSim)

#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "plane.urdf" ))

#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/octopus_simple_2_troubleshoot.urdf" ) ) 

#default urdf to troubleshoot is octopus_simple_2_troubleshoot.urdf

#plane = p.loadURDF( "home/roman/Documents/RoboLab/bullet3/examples/pybullet/gym/pybullet_data/romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf")

#plane = p.loadURDF( os.path.join( path_to_xml_file, "python_scripts_edit_urdf/output2.urdf" ) )

#print(os.path.join( path_to_xml_file, "python_scripts_edit_urdf/output2.urdf)
 
plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf" ) )


#blockPos = [0,0,3]
#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "cube_small.urdf" ) , basePosition = blockPos    )

#p.resetSimulation()
#robotId = p.loadURDF( os.path.join(pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/my_robot.urdf" ) )

##################velocity??????? 2pm jan 6, 2019
#angles = p.calculateInverseKinematics(bodyUniqueId=plane, endEffectorLinkIndex=0, targetPosition = [-0.5 , 0, 0  ], solver=p.IK_DLS )

############uncomment this ##set target position
#p.setJointMotorControl2(bodyUniqueId=plane, jointIndex=0, controlMode=p.POSITION_CONTROL,targetPosition=0.5 )

#state=p.getJointState(bodyUniqueId=plane, jointIndex=0, physicsClientId=clientId)

#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.POSITION_CONTROL,targetPosition=0.5, 
##currentPosition=state[0] , 
#force=1) #move link to target position


#uncomment this #lockJoint
#p.createConstraint(parentBodyUniqueId=plane, parentLinkIndex=-1, childBodyUniqueId=plane, childLinkIndex=0, jointType=p.JOINT_FIXED, jointAxis=[0,0,0], parentFramePosition=[2,2,2], childFramePosition=[2,2,2]  )

#########LOCK joint with force
#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=10000)#lock joint with large force

#p.setJointMotorControl2(bodyIndex=plane, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=1, force=0) #free all joint constraints with force 0



angles = p.calculateInverseKinematics2(bodyUniqueId=plane, endEffectorLinkIndices=[3], targetPositions= [[ 0, 0.1, 0.1]], solver=p.IK_DLS) #the x coordinate is zero because it is a planar robot at the moment #the x coordinate bounds are (-6,6), the y coordinate bounds are (-4,8)

p.setJointMotorControlArray(bodyUniqueId=plane, jointIndices=[0,1,2,3], controlMode=p.POSITION_CONTROL, targetPositions=angles, forces = [1000,1000,1000,1000])

print(angles)

while(1):
	#print(angles)
	(  ( l1, l2, l3, l4, l5, l6 ),  ) = p.getLinkStates(bodyUniqueId=plane, linkIndices=[3] )  #returns tuple
	end_effector_world_position = list(l1) #convert tuple to list
	#l = p.getLinkStates(bodyUniqueId=plane, linkIndices=[3] )
	print()
	print(end_effector_world_position)
	a=2	
	#state = p.getJointState(bodyUniqueId=plane, jointIndex = 1, physicsClientId= clientId) 
	#print(state)
	#print(state[0])
#	print(angles)
	
	

#while(1):
#	print(444444444)
