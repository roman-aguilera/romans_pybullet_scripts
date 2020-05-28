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
        import pybullet_envs
        self.env = gym.make("CartPoleBulletEnv-v1", renders=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
	
class ReacherEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        #self.env = gym.make( choose_env_for (env_config.worker_index, env_config.vector_index))
        import pybullet_envs
        self.env = gym.make("ReacherBulletEnv-v0", render=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)


register_env("cartpolebulletenv", lambda config: MultiEnv(config))
register_env("reacherbulletenv", lambda config: ReacherEnv(config))


#trainer = ppo.PPOTrainer(config=config, env="cartpolebulletenv")
trainer = ppo.PPOTrainer(config=config, env="reacherbulletenv")

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



