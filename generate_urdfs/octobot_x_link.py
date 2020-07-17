import pybullet as p
import time
import pybullet_data #helps locate urdf file
import os
import sys
import numpy as np
import pdb
from einsteinpy import coordinates


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
#plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf" ) )
plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/octopus_generated_20_links.urdf" ) )
#plane = p.loadURDF("octopus_generated_3_links.urdf")
#plane = p.loadURDF( os.path.join(os.getcwd(), "output2.urdf") )

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


while(1):
#	print(angles)
	(  ( l1, l2, l3, l4, l5, l6 ),  ) = p.getLinkStates(bodyUniqueId=plane, linkIndices=[3] )  #returns tuple
	end_effector_world_position_cartesian = list(l1)
	#l = p.getLinkStates(bodyUniqueId=plane, linkIndices=[3] )
	
	#pint out EE cartesian coordinates
	print("The end effector world position in Cartesian coordinates is: [x,y,z] ")
	print(type(end_effector_world_position_cartesian))
	print( end_effector_world_position_cartesian )
	
	#print out EE polar coordinates
	
	print("The end effector posiaiton in Polar coordinates are: [radius, theta, phi]")
	#print(end_effector_world_position_cartesian)
	end_effector_world_position_polar = coordinates.utils.cartesian_to_spherical_novel(end_effector_world_position_cartesian[0], end_effector_world_position_cartesian[1], end_effector_world_position_cartesian[2] ) 	
	print( type(end_effector_world_position_polar) )
	print(end_effector_world_position_polar)
	
	a=2	
	state = p.getJointStates(bodyUniqueId=plane, jointIndices = [0,1,2,3], physicsClientId= clientId) 
	print("The state of the joints are")
	print(type(state))
	print(len(state))
	print(len(state[0]))
	print(state)
	#print(state[0])
#	print(angles)
	
	

#while(1):
#	print(444444444)
