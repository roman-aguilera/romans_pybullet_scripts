import gym

#got this from # https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/help
# 
import pybullet_envs.bullet.minitaur_gym_env as e
env = e.MinitaurBulletEnv(render=True)


class MultiEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = gym.make(
            choose_env_for(env_config.worker_index, env_config.vector_index))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

register_env("multienv", lambda config: MultiEnv(config))

trainer = ppo.PPOTrainer(env="my_env")


