
'''
Pythonscropts of relevance is found in 


Emple Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
'''


'''

Actual implementation of save model for cartpole balancing

rllib rollout ~/ray_results/PPO_CartPole-v0_2020-05-26_01-37-58fg_p0r94/checkpoint_1/checkpoint-1 --run PPO --env CartPole-v0 --steps 1000000 --out rollouts.pkl




Rollout of cartpole bullet env

```
./rollout.py /home/roman/ray_results/PPO_multienv_2020-05-26_17-36-04dsfcgant/checkpoint_581/checkpoint-581 --run PPO --env cartpolebulletenv --steps 1000000 --out rollouts.pkl --no-render 

```







