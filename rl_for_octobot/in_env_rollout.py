

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import gym

#ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0
config["num_cpus_per_worker"] = 6
config["eager"] = False

class WalkerEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("Walker2DBulletEnv-v0", render=True)#~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_locomotion_envs.py
        self.debug = True
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.dts_taken_so_far = 0 # used for debugging number of timesteps taken in one episode
    def reset(self, initial_joint_state):
        self.env.reset() #
        joint_index = 0
        for j in self.env.env.robot.ordered_joints:
            joint_index +=2
            j.reset_current_position(initial_joint_state[joint_index], initial_joint_state[joint_index+1]) #reset joint state to desired position and velocity
            #j.reset_current_position(0.0,0.0) #reset joint state to desired position and velocity
        #return self.env.reset()
        self.env.step(self.env.action_space.sample()*0.0) #let robot fall without taking any action for 1 timestep, return usual env paratmeters
        self.dts_taken_so_far = 1
        return self.env.env.robot.calc_state() #return state of robot
    def step(self, action):
        input("Press Enter  .....")
        print("Colisions for feet:                                             ", self.env.env.robot.calc_state()[20], "   ", self.env.env.robot.calc_state()[21]) #returns states, last 2 numbers inticate whther foot is in contact with ground
        self.dts_taken_so_far += 1
        if self.debug:
            print("Time elapsed in episode: ", self.dts_taken_so_far * self.env.env.scene.dt)
            print("Number of dt's taken in episode: " , self.dts_taken_so_far)
        return self.env.step(action)

from ray.tune.registry import register_env
register_env("walkerbulletenv", lambda config: WalkerEnv(config))

#trainer = ppo.PPOTrainer(config=config, env="walkerbulletenv")

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config['num_workers'] = 0

ckpt_path = "/home/roman/ray_results/PPO_walkerbulletenv_2020-05-28_22-26-581q14o5cv/checkpoint_991/checkpoint-991" #path to saved policy
agent = ppo.PPOTrainer(config, env="walkerbulletenv")
agent.restore(ckpt_path)                             # restore agent (policy) from checkpoint
policy = agent.workers.local_worker().get_policy()   # get the policy

import pybullet_envs
env_instance = gym.make("Walker2DBulletEnv-v0", render=True)

import numpy as np

from pydub import AudioSegment #for sound output
from pydub.playback import play
song = AudioSegment.from_mp3("Ding Sound Effect.mp3")


#createcsv for joint states
import os
path_to_link_7_walker_timeseries_data_csv_file = os.path.join(os.getcwd(), "link_7_walker_timeseries_data.csv") 
link_7_walker_timeseries_data_csv_file = open(file=path_to_link_7_walker_timeseries_data_csv_file, mode='x') # , newline='') # https://docs.python.org/3/library/functions.html#open #open with exclusive creation
import csv
filewriter_for_link_7_walker = csv.writer(link_7_walker_timeseries_data_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL) 
filewriter_for_link_7_walker.writerow(['thigh_joint_torque', 
				'leg_joint_torque',
				'foot_joint_torque',
				'thigh_left_joint_torque',
				'leg_left_joint_torque',
				'foot_left_joint_torque',

				'thigh_joint_position',
				'leg_joint_position',
				'foot_joint_position',
				'thigh_left_joint_position',
				'leg_left_joint_position',
				'foot_left_joint_position',

				'thigh_joint_velocity',
				'leg_joint_velocity',
				'foot_joint_velocity',
				'thigh_left_joint_velocity',
				'leg_left_joint_velocity',
				'foot_left_joint_velocity',

				'time'])


#start from beginning of rollout
next_obs = env_instance.reset()

for num,joint_x in enumerate(env_instance.env.ordered_joints):
        num, joint_x, joint_x.joint_name #bookkeeping for joints

for body_part_x in env_instance.env.parts.values():
	body_part_x.current_position()
	body_part_x.current_orientation()

for j in env_instance.env.robot.ordered_joints:
	j.reset_current_position(0.0, 0.0)
	j.enable_sensor()

for body_part_x in env_instance.env.parts.values():
        body_part_x.current_position()
        body_part_x.current_orientation()

next_obs=env_instance.env.robot.calc_state()
time = 0
time_per_gym_step = env_instance.env.scene.dt 

distance_traveled = 0
num_distance_resets = 0
distance_reset_increments = 30

env_instance.env.robot.walk_target_x = 1000
env_instance.env.robot.body_xyz
next_obs

for time_steps_in_sim in range (90000):
	a, b, c =  policy.compute_actions([next_obs]); next_obs, reward, done, info =  env_instance.step( a.reshape(a.size,) ); env_instance.env.robot.walk_target_dist; next_obs; time += time_per_gym_step
	
	data_list=[] #this is the lsit we add to the csv file
	
	for n,j in enumerate(env_instance.env.robot.ordered_joints):
		action_torque = env_instance.env.robot.power * j.power_coef * float(np.clip(a.reshape(a.size,)[0], -1, +1)) #add joint torque
		data_list.append(action_torque) #add joint toques

	for n,j in enumerate(env_instance.env.robot.ordered_joints):
		data_list.append(j.get_position()) #add joint positions

	for n,j in enumerate(env_instance.env.robot.ordered_joints):
		data_list.append(j.get_position()) #add joint velocities

	data_list.append(time) #append timestamp

	filewriter_for_link_7_walker.writerow(data_list)
	if env_instance.env.robot.walk_target_dist < 1e3-distance_reset_increments:
		env_instance.env.robot.walk_target_x += distance_reset_increments
		distance_traveled += distance_reset_increments
	print("Distance Traveled: ", distance_traveled)
	print("Distance to target : " , env_instance.env.robot.walk_target_dist)
	if env_instance.env.robot.body_xyz[2] < 0.7: #robots torso is low enough that it might have fallen on knees
		play(song) #ding when fallen
		break
	#if done == True:
	#	break







