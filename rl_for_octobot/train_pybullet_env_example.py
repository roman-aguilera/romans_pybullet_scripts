


import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import gym


ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["num_cpus_per_worker"] = 6
config["eager"] = False


#set env to minitaur
#got this from # https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/help
#import pybullet_envs.bullet.minitaur_gym_env as e
#env = e.MinitaurBulletEnv(render=True)

#import pybullet_envs.gym_manipulator_envs as e2



#import pybullet_envs 
#envs are registered during import. 
#envs are found in ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/__init__.py


class MultiEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
	#self.env = gym.make( choose_env_for (env_config.worker_index, env_config.vector_index))
        import pybullet_envs #/home/roman/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/__init__.py
        self.env = gym.make("CartPoleBulletEnv-v1", renders=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
	
class ReacherEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("ReacherBulletEnv-v0", render=False) #check ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_manipulator_envs.py
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

class PusherEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("PusherBulletEnv-v0", render=False) # check ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_manipulator_envs.py
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

class ThrowerEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("ThrowerBulletEnv-v0", render=False) # check ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_manipulator_envs.py
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

class StrikerEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("StrikerBulletEnv-v0", render=False) # check ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_manipulator_envs.py
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

class WalkerEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("Walker2DBulletEnv-v0", render=False)#~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_locomotion_envs.py
        # if you want to modify the plane, the sdf plane file is loaded in ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/scene_stadium.py
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

class WalkerPlaneChangingEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make("Walker2DBulletEnv-v0", render=False)#~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_locomotion_envs.py
        # robot is found in ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/robot_locomotors.py : Walker2D
# If you want to modify the ground plane, the openAI gym env loads the the sdf plane in this part of the code ~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/scene_stadium.py
#To modify the floor plane here (for the env), during th RL lib env initialization, do the following:
        #self.env.reset() 
        #self.env.env._p.removeBody(0) #take out floor plane 
        #self.env.env.stadium_scene.ground_plane_mjcf = env_instance.unwrapped._p.loadSDF(os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf"))  #and new plane
        #self.env.reset() #reset (just in case. check if gym works after this is commented out)
        
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)




#NEED TO ADD THESE verified envs:
#import pybullet_envs
#env = gym.make("Walker2DBulletEnv-v0", render=True)#~/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_envs/gym_locomotion_envs.py


register_env("cartpolebulletenv", lambda config: MultiEnv(config))
register_env("reacherbulletenv", lambda config: ReacherEnv(config))
register_env("pusherbulletenv", lambda config: PusherEnv(config))
register_env("throwerbulletenv", lambda config: ThrowerEnv(config))
register_env("strikerbulletenv", lambda config: StrikerEnv(config))
register_env("walkerbulletenv", lambda config: WalkerEnv(config))

#trainer = ppo.PPOTrainer(config=config, env="cartpolebulletenv")
#trainer = ppo.PPOTrainer(config=config, env="reacherbulletenv")
#trainer = ppo.PPOTrainer(config=config, env="pusherbulletenv")
#trainer = ppo.PPOTrainer(config=config, env="throwerbulletenv")
#trainer = ppo.PPOTrainer(config=config, env="strikerbulletenv")  #was not able to be trained (had to remove ['joint'] from a dictionary call somwhere in the env)
trainer = ppo.PPOTrainer(config=config, env="walkerbulletenv") 

#register_env("octoenv", lambda config: OctoEnv(config))
#trainer = ppo.PPOTrainer(config=config, env="octoenv")


for i in config:
    print(i)
input("press enter to continue")
# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 10 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)



