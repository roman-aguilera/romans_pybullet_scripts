

Pythonscrpts of relevance is found in 

```
~/go_to_project
```

```
Emple Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
```


Actual implementation of save model for cartpole balancing
```
rllib rollout ~/ray_results/PPO_CartPole-v0_2020-05-26_01-37-58fg_p0r94/checkpoint_1/checkpoint-1 --run PPO --env CartPole-v0 --steps 1000000 --out rollouts.pkl
```


To TRAIN, run the script
```
python pybullet_env_example.py
```

Rollout of cartpole bullet env

```
python rollout.py /home/roman/ray_results/PPO_multienv_2020-05-26_17-36-04dsfcgant/checkpoint_581/checkpoint-581 --run PPO --env cartpolebulletenv --steps 1000000 --out rollouts.pkl --no-render 

```

Rollout of reacher bullet env
```
python rollout.py /home/roman/ray_results/PPO_reacherbulletenv_2020-05-27_23-48-305gb2g0nd/checkpoint_141/checkpoint-141 --run PPO --env reacherbulletenv --steps 1000000 --out rollouts.pkl --no-render 

```

Rollput of pusher bullet env
```
python rollout.py /home/roman/ray_results/PPO_pusherbulletenv_2020-05-28_02-02-04ci_56jrd/checkpoint_141/checkpoint-141 --run PPO --env pusherbulletenv --steps 1000000 --out rollouts.pkl --no-render
```

Rollout of thrower bullet env
```
python rollout.py /home/roman/ray_results/PPO_throwerbulletenv_2020-05-28_03-17-08rrgno8yp/checkpoint_291/checkpoint-291 --run PPO --env throwerbulletenv --steps 1000000 --out rollouts.pkl --no-render
```

Rollout of walker
```
python rollout.py /home/roman/ray_results/PPO_walkerbulletenv_2020-05-28_22-26-581q14o5cv/checkpoint_991/checkpoint-991 --run PPO --env walkerbulletenv --steps 1000000 --out rollouts.pkl --no-render

```




UPDATE following

Rollout of striker bullet env
```
python rollout.py /home/roman/ray_results/PPO_reacherbulletenv_2020-05-27_23-48-305gb2g0nd/checkpoint_141/checkpoint-141 --run PPO --env strikerbulletenv --steps 1000000 --out rollouts.pkl --no-render
```



