

the urdf files that are loaded by pybullet in the `roman_playful` environment are found in 

``` /home/roman/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_data/romans_urdf_files/octopus_files/python_scripts_edit_urdf  ```


you can test a current urdf file by running the command

``` python octobot_x_link.py ```

to test another URDF, change the directory that is fed into the p.loadURDF() method in `octobot_x_link.py`


to generate a new UDRF with a wanted number of links, go into the `/home/roman/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_data/romans_urdf_files/octopus_files/python_scripts_edit_urdf/create_urdf_old2.py` and change the 'number_of_links' to however many you want then run the script from the command line

`python create_urdf_old2.py`
a new URDF and XML file for the robot will be created in the current directory. If a URDF/XML file of a similar name already exists, it will be overwritten

the octopus gym environment is located in `...pybullet_envs/octobot_env.py`
the `...pybullet_envs.__init__` file was edited with the following lines following the definition of the `register()` function
```
# -----------octopus------------

register(
    id='OctopusArmBulletEnv-v0', #env_instance = gym.make('OctopusArmBulletEnv-v0', renders=True)
    entry_point='pybullet_envs.octopus_env:OctupusEnv',
    max_episode_steps=1000,
    reward_threshold=20000.0,
)

```



