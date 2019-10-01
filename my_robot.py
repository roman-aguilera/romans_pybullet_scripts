import pybullet as p
import time
import pybullet_data
import os
import sys

#cid = p.connect(p.SHARED_MEMORY)
#cid = p.connect(p.DIRECT)
cid = p.connect(p.GUI)

if (cid<0):
        p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0,0,-10)
useRealTimeSim = 0
if useRealTimeSim == 0:
	p.setTimeStep(1. / 500)

p.setRealTimeSimulation(useRealTimeSim) # either this
plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))

#p.resetSimulation()
robot_start_position = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

#rs = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"romans_urdf_files/octopus_files/my_robot.urdf"), basePosition=robot_start_position, baseOrientation=robot_start_orientation )

#rs = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"romans_urdf_files/send_to_sean/robosim_UCSB.urdf") , [2,0,1],flags=8)

octopus2 = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"romans_urdf_files/octopus_files/octopus.urdf"), basePosition=robot_start_position, baseOrientation=robot_start_orientation )

#octopus2 = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"romans_urdf_files/octopus_files/octopus2.urdf"), basePosition=robot_start_position, baseOrientation=robot_start_orientation, useFixedBase=0 )

number_of_joints = p.getNumJoints(octopus2, cid)
print("number of ojoints type", type(number_of_joints))
joint_info = p.getJointInfo(octopus2, cid)
print("Number of Joints: ", number_of_joints)
print("Joint Information: ", joint_info)
p.getJointInfo(octopus2 , cid)

joint_indices = [i for i in range(number_of_joints)]
motor_torques = [0] * number_of_joints #list of zeros
motor_forces = [1] * number_of_joints #list of zeros
motor_torques[0] = 0
print("motor  forces : ", motor_forces)
#motor_velocities[0] = 3
print("joint indices: ", len(joint_indices), "motor velocities: ", len(motor_torques) )
#motor_velocities[1] = 6 #edit the individual motor velocity
#motor_forces = [0] * number_of_joints #list of zeros
p.setJointMotorControlArray(bodyIndex=octopus2, jointIndices=joint_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=motor_torques, forces=motor_forces)
while(True):
	#time.sleep(10)
	if useRealTimeSim==0:
		p.stepSimulation()
	else:
		#time.sleep(1./240.)
		print()
	
	#p.setJointMotorControlArray(bodyIndex=octopus2, jointIndices=joint_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=motor_torques, forces=motor_forces)#targetTorques=motor_torques, forces = motor_forces )	
	#bodyUniqueId
	#print(p.getNumJoints(octopus2_, cid), p.getJointInfo(octopus2_, cid))


