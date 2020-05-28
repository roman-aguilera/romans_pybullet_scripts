"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import einsteinpy
from einsteinpy import coordinates 
from scipy.spatial import distance
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Discrete, Box, Dict
print ("gym.spaces imported")
import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

tf = try_import_tf()


"""pybullet packages import start"""
import pybullet as p
import time
import pybullet_data #helps locate urdf file
import os #helps find paths for current python packages being used
import sys
import numpy as np
import pdb #python debugger


""" end pybulet packages import"""


class OctoBotEnv(gym.Env):
	"""Example of a custom env in which you have to walk down a corridor.

	You can configure the length of the corridor via the env config."""
####INIT
	def __init__(self, config):
	
        #connect to pybullet ohysics server
		if config["env_render"] == 1:
			clientId = p.connect(p.GUI)
		else: 
			clientId = p.connect(p.DIRECT)
		#clientId = p.connect(p.GUI)
	#clears all urdf (Unified Robot Description Format) aka the 3D model of our robot 
		p.resetSimulation()
	#set gravity
		p.setGravity(0,0,-10)
	#determine if our simulation will: (1) run in real time or if,  (0) we need to manually step (a fixed timestep) when we take a atep in out RL gym envorionment
		useRealTimeSim = 1
		p.setRealTimeSimulation(useRealTimeSim)
	#find directory of file where our urdf is located
		print( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf" ) )
	#load our urdf 
		plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf" ) )
	#octobot hyperparameters	
		self.total_number_of_links = config["total_number_of_links"]
		self.number_of_free_joints = 1 # config["number_of_free_joints"]
		self.total_number_of_joints = self.total_number_of_links
		
        #torque limits (originally same for each joint but we can change that later)
		self.torque_limit_min = -1000
		self.torque_limit_max = 1000

	#joint state limits (originally same for each joint (also same for joint positions and velocities) but we can change that later)
		self.joint_states_min = 1000
		self.joint_states_max = 1000

	#initial distance to goal vector (in polar coordinates) aka randomaly assign where we want our end effector to move to
		self.distance_to_goal_epsilon = 0.5 #how close (in radial distance) does the end effector need to be to the goal in order for us to consider the goal to be "reached"
		self.max_distance_to_goal = 8 # trying to reach far goals will take a long time to obtain rewards
		self.max_radial_distance_from_base = 8 # helps make sure goal is within reachable workspace

	# Sample goal points (for case of 2D arm)
	#For reference "sample points uniformly within a circle"        
		#https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly  
		# takeaway: account for the fact that radius doubles witin a sphere
		R = self.max_radial_distance_from_base_to_goal
		self.radial_distance_from_base_to_goal = R * math.sqrt( random.random() ) 	# radial distance from base to goal
		self.angle_from_base_to_goal_theta = random.random() * 2 * math.pi		#base to goal theta
		centerX = 0 #in pybullet, this is actually th y coordinate
		centerY = 2 #in pybullet, this is actually the z coordinate
		x = centerX + r * math.cos(self.angle_from_base_to_goal_theta)
		y = centerY + r * math.sin(self.angle_from_base_to_goal_theta)
		
		#theta is given for 2D case but not relevant input to NN			#base to goal phi
		if x>=0:
			self.angle_from_base_to_goal_phi = math.pi/2  # +y cartesian coord gives +theta, -y gives -theta
		else: 
			self.angle_from_base_to_goal_phi = -math.pi/2

        #Sample points (for case of 3D arm)
                #sample points uniformly within a sphere
                #https://stackoverflow.com/a/5408843    
                # takeaway: account for the fact that volume 
                        #phi = random.uniform(0,2pi)
                        #costheta = random.uniform(-1,1)
                        #u = random.uniform(0,1)
                        #theta = arccos( costheta )
                        #r = R * cuberoot( u )
                        #x = r * sin( theta) * cos( phi )
                        #y = r * sin( theta) * sin( phi )
                        #z = r * cos( theta )
		
		
		
		#self.distance_to_goal = random.uniform( 0, self.distance_to_goal_threshold ) #
		#self.angle_to_goal_theta = random.uniform(0, 2*3.14158 )        #elevation angle 
		#self.angle_to_goal_phi = random.uniform(0, 2*3.14158 )          #azimuthal angle
	
		#self.radial_distance_from_base_to_goal = r      		#random.uniform( 0, self.max_radial_diastance_from_base ) 
		#self.angle_from_base_to_goal_theta = theta      		#random.uniform(0, 2*3.14158 )        #elevation angle 
		#self.angle_from_base_to_goal_phi = (math.pi)/2     #2d arm case  	#random.uniform(0, 2*3.14158 )          #azimuthal angle
		
		
		
	#get xyz potition of goal
		self.cartesianPositionOfGoal = [0, x, y]
	#get cartesian vector from base to EE (end effector)
		(  ( l1, l2, l3, l4, l5, l6 ),  ) = p.getLinkStates(bodyUniqueId=plane, linkIndices=[ self.total_number_of_links - 1 ] )  #returns tuple
		self.vector_from_base_to_EE_cartesian = list(l1) #vector from base to EE
			
	#get cartesian vector from EE to goal	
		self.vector_from_EE_to_goal_cartesian = np.subtract( self.vector_from_base_to_goal, self.vector_from_base_to_EE ) #returns nupy array
	
	#get disance from EE to goal
		#self.radial_distance_from_EE_to_goal = np.linalg.norm(self.vector_from_EE_to_goal)  # returns numpy float.64 	#random.uniform( 0, self.distance_to_goal_threshold ) 
	
	#get polar vector from EE to goal
		self.vector_from_EE_to_goal_polar = coordinates.utils.cartesian_to_spherical_novel( self.vector_from_EE_to_goal_cartesian[0] , self.vector_from_EE_to_goal_cartesian[1] , self.vector_from_EE_to_goal_cartesian[2]  )
	
	# extract each polar coordinates (r, theta, phi) from EE_to_goal_vector to gym environment			
		self.radial_distance_from_base_to_goal = self.vector_from_EE_to_goal_polar[0] 	#random.uniform( 0, self.max_radial_diastance_from_base ) 
		self.angle_from_base_to_goal_theta = self.vector_from_EE_to_goal_polar[1]    	#random.uniform(0, 2*3.14158 )        #elevation angle 
		self.angle_from_base_to_goal_phi = self.vector_from_EE_to_goal_polar[2]		#random.uniform(0, 2*3.14158 )          #azimuthal angle
		
	#turn spherical position of goal to cartesian coordinates (for visualization) 
		#self.position_of_goal = einsteinpy.utils.coord_transforms.SphericalToCartesian_pos( self.radial_distance_from_base_to_goal, self.angle_from_base_to_goal_theta, self.angle_from_base_to_goal_phi )
		self.vector_from_base_to_goal_cartesian = coordinates.utils.spherical_to_cartesian_novel( self.radial_distance_from_base_to_goal, self.angle_from_base_to_goal_theta, self.angle_from_base_to_goal_phi )

		#self.distance_to_goal = random.uniform( 0, self.distance_to_goal_threshold ) #
		#self.angle_to_goal_theta = random.uniform(0, 2*3.14158 )	#elevation angle 
		#self.angle_to_goal_phi = random.uniform(0, 2*3.14158 )		#azimuthal angle
 
		
        # This function generates all binary strings of length n with k number of bits(... akak k number of ones) 
	# (example: n=3, k=2, kbits(n,k) gnerates ['110', '101', '011'])
        # https://stackoverflow.com/questions/1851134/generate-all-binary-strings-of-length-n-with-k-bits-sets ### 
		import itertools
		def kbits(n, k):
			result = []
			for bits in itertools.combinations(range(n), k):
				s = ['0'] * n
				for bit in bits:
					s[bit] = '1'
			result.append(''.join(s))
			return result

        #use off the shelf kbits function to generate all possible list of combinations (for free joints) 
		self.list_of_combinations = kbits(self.total_number_of_links, self.number_of_free_joints)
			
#CREATE OBSERVATION SPACE	
        #how to use Box object type
        #examples of how to use Box are found in Documents/RoboLab/roman_playful/gym/gym/spaces/box.py 
		self.observation_space = Dict( {
			"radial distance to goal" : Box( low = 0, high = self.distance_to_goal_threshold, dtype = np.floatt32 ) , 
			"angle (polar,theta) to goal" : Box( low = 0, high = 2*3.15, dtype = np.float32 ) ,
			"angle (azimuthal) to goal" : Box( low = 0, high - 2*3.15, dtype = np.float32 ), 
			"joint states" : Box( low = self.joint_states_min*np.ones( self.total_number_of_joints), high = self.joint_states_max*np.ones(self.total_number_of_joints) , dtype=np.float32 ) ,
		} )

#CREATE ACTION SPACE            
        #how to use Discrete object type
        #Documents/RoboLab/roman_playful/gym/gym/spaces/discrete.py
		self.action_space = Dict({ 
			"decide which joints to move" : Discrete( self.number_of_free_joints ) , #chose an index that specifies which binary string will be used to decide whoch joints are locked and which joints are free
			"decide torques on each free joint" : Box( low = self.torque_limit_min*np.ones( self.number_of_free_joints), high = self.torque_limit_max*np.ones(self.number_of_free_joints) , dtype=np.float32 ) ,   #decided torques to implement on each free joint
		})
	
	"""end pybullet contribution"""


	"""
	self.end_pos = config["corridor_length"]
	self.cur_pos = 0

	self.action_space = Discrete(2)
	self.observation_space = Box(0.0, self.end_pos, shape=(1, ), dtype=np.float32)
	"""
#### THINGS TO NOte ABOUT 4 LINK URDF
	####
	####
	#### 



#### RESET
	def reset(self):
	### begin pybullet stuff
		p.resetSimulation() #
		plane = p.loadURDF( os.path.join( pybullet_data.getDataPath(), "romans_urdf_files/octopus_files/python_scripts_edit_urdf/output2.urdf" ) )  
	### end pybullet stuff
		
		self.desired_position = [0, random.uniform(-6,6), ] #######REWWI
		dictionary_of_states = { "radial distance to goal" : ,
					"angle (polar,theta) to goal" : ,
					"angle (azimuthal) to goal" : ,
					"joint states" :
					}
		
		return [self.cur_pos]
#### STEP
	def step(self, action):
		print("the action taken is")
		print(action)
		print(type(action))
		#action["decide which joints to move"]
		print(action)
	#create assertions for action space
		assert action["decide which joints to move"] in list[range(0,self.number_of_joints)] #, action["decide which joints to move"]
	# decide what joints to lock and which to leave free	
		list_index = action["decide which joints to move"]
		self.list_of_combinations[ list_index ] #choose which cobination of joints to unlock (represented as string)
		for i in self.list_of_combinations[ list_index ]:
			print(action["decide which joints to move"])
	
	#sdf
		
	# perform actation of joints (create for loop to step through simulation) (input a torque on joints for certain amount of loops)
		
	
		#if action == 0 and self.cur_pos > 0:
		#	self.cur_pos -= 1
		#elif action == 1:	
		#	self.cur_pos += 1
		#done = self.distance_to_goal >= self.end_pos
		
		
		#GET END EFFECTOR POSITION #side note : base position is (0,0,2)
		(  ( l1, l2, l3, l4, l5, l6 ),  ) = p.getLinkStates(bodyUniqueId=plane, linkIndices=[self.total_number_of_links - 1] )
		end_effector_world_position = np.asarray([l1]) #convert tuple to numpy array (so that we can compute euclidean distance later using scipy)
		distance_to_goal_vector = self.desired_xyz_position_of_end_effoctor - self.current_xyz_position_of_end_effector
		print("distance to goal vector is:") print( distance_to_goal_vector )
		observation["distance_to_goal"], observation["angle_theta_to_goal"], observation["angle_phi_to_goal"] = einsteinpy.utils.coord_transforms.CartesianToSpherical_pos( distance_to_goal_vector )
		
		#GET JOIN 
		done  = distance_to_goal <= self.distance_to_goal_threshold
		#return [self.cur_pos], 1 if done else 0, done, {}
		
		if self.time_to_reach_goal > 5 self.distance_to_goal < self.distance_to_goal_epsilon:
			done = True
		
		info = {}
		return observation, reward, done, info
#### RENDER
	#def render(self, mode='human', close=False):
	#	pass


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    tune.register_env("octobot", lambda config: OctoBotEnv(config))
    ray.init(num_cpus=4)
    
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": "octobot", #OctoBotEnv,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": 1e-2, #grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
                "env_render" : 0,
                "total_number_of_links" : 4,
                "number_of_free_joints" : 1,
            },
        },
    )
