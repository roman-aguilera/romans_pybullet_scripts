


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


policy = CustomPolicy(env.observation_space, env.action_space, {})
workers = WorkerSet(
    policy=CustomPolicy,
    env_creator=lambda c: gym.make("CartPole-v0"),
    num_workers=10)
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

