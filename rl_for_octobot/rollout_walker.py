


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
    #def set_joint_states_from_disturbance_profile(self, initial_joint_state):
    #    joint_index = 0
    #    for j in self.env.env.robot.ordered_joints:
    #        joint_index +=2
    #        j.reset_current_position(initial_joint_state[joint_index], initial_joint_state[joint_index+1]) #reset joint state to desired position and velocity


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

#for i in range(1000):
#    print("Episode : ", i)
#    done = False
#    next_obs = env_instance.reset()
#    while not done:
#        a, b, c =  policy.compute_actions([next_obs])
#        next_obs, reward, done, info =  env_instance.step( a.reshape(a.size,) )    
#env_instance.close()



#################################### start mesh algorithm initialization
import numpy as np
Mesh_States = np.concatenate( (np.ones((1,20))*np.inf,  np.zeros((1,20)) ) , axis=0 )  #using first 20 states of walker  
#if using cvxpy for distances, this line should be Mesh_States = np.concatenate( ( np.ones((30,1))*np.inf, np.zeros((30,1)) ) , axis=1)
current_state_index = 1 # initialize current state index
total_number_of_states = 2
from sklearn.neighbors import NearestNeighbors
#import numpy as np

nbrs = NearestNeighbors(n_neighbors=1).fit(Mesh_States[1:,:]) #create KD-tree/ball-tree of Mesh using scikitlearn #omit first(failure) state

distance_threshold=15

#import cpvpy as cp # import function to calculate distance
#from .state_distance_calculator.py import is_distance_threshold_exceeded # import function to calculate distance 
#################################### end mesh algorithm initialization

##################################### start creation of disturbance profile
print("creating disturbance profile... ")
number_of_perturbable_states = 12
mean_of_multivariate_distribution = np.zeros(number_of_perturbable_states).reshape(number_of_perturbable_states,)

covariance_matrix = np.identity(number_of_perturbable_states) #empty n-by-n array
covariances = np.ones((number_of_perturbable_states,1)) #array of covariances
standard_deviations = np.sqrt(covariances)
np.fill_diagonal(a=covariance_matrix, val=covariances ) #fills in diagonals of (initially empty) covariance matrix with array of covariances

from scipy.stats import multivariate_normal
var = multivariate_normal(mean=mean_of_multivariate_distribution.tolist(), cov=covariance_matrix.tolist()) #alternate method ##var = multivariate_normal(mean=number_of_perturbable_states*[0.0], cov=np.identity(number_of_perturbable_states).tolist()) #create multivariate distribution for joint states to perturb
#var.pdf() #this tels you the porbability of having the value in that index


#disturbances = np.mgrid[ [slice(mean_of_multivariate_distribution[num]-3*standard_deviations[num], covariances[num], 0.5) for num,i in enumerate(mean_of_multivariate_distribution.reshape(3,1)) ] ]  #returns a np array with dimenstions (num_perturbable_states, upper[0]-lower[0]]/granularity[0] , ..., upper[num_pert_states-1]-lower[num_pert_states-1] / granularity[num_pert_states-1] ) # here, shape = (12,4,4,4,4,4,4,4,4,4,4,4,4)

disturbances = np.mgrid[ [slice(mean_of_multivariate_distribution[num]-2*standard_deviations[num], mean_of_multivariate_distribution[num]+2*standard_deviations[num],1.0) for num,i in enumerate(mean_of_multivariate_distribution.reshape(number_of_perturbable_states,1)) ] ]

#disturbances = np.meshgrid(*l)

print("creating probability desnity matrix of disturbances ... ")
probabilities = var.pdf( np.moveaxis(disturbances,0,-1) ) # make first axis the last, other axes retain their relative order # shape goes from (12,4,4,4,4,4,4,4,4,4,4,4,4) to (4,4,4,4,4,4,4,4,4,4,4,4,12) #get the probablilities for each possible disturbance of the state, based on our multivariate gaussian model probabilities are in shape of (4,4,4,4,4,4,4,4,4,4,4,4)
probabilities.sum(axis=None) #sum of all probabilities
probabilities = probabilities/probabilities.sum(axis=None) # normalize disturbnce probability distribution
###################################### end creation of disturbance profile

###################################### start initializations of T_det, the derteministic state transition matrix
init_numstates=2
num_controllers=1
shape_of_disturbance_profile = probabilities.shape
shape_of_T_det = (init_numstates, num_controllers)+(shape_of_disturbance_profile) #append tuples
print("initializing Deterministic transition matrix...")
T_det = np.zeros(shape=shape_of_T_det, dtype=int)

###################################### end inittializations od T_det
while(1): #runn though all states
    print("Meshing Progress : ", current_state_index , "/", total_number_of_states )
    if current_state_index == total_number_of_states: # stop if we have visited all states in mesh table
        break
    for controller_index in range(num_controllers):
        
        it = np.nditer(probabilities, flags=['multi_index']) #helps keep track of index when we iterate over array of disturbances
        for x in it: # we cver all the possible disturbances # x is the value in the disturbance probability array, it.multi_index is a tuple representing the current index value in the disturbance probability array
            probability_of_disturbance = x
            perturbation = np.array([])
            for i in range(0, number_of_perturbable_states):	                           #create single perturbation
                perturbation = np.append(perturbation, disturbances[(i,)+it.multi_index])  #it.multi_index is a tuple of current index value in the distrubance probability array
    
            #start simulation
            #print("Meshing Progress : ", current_state_index , "/", total number of states )
            #if current_state_index = total_number_of_states: # stop if we have visited all states
            #    break
            
            number_of_remaining_simulation_steps = 300  #### amount of simulation steps we want to run until we log into our state trasition matrix 
                                                        #### [(1 dt)= (1 step in simulation) = (env.instance.emv.scene.dt=0.0165) 
                                                        #### = (env_instance.env.scene.frameskip = 4) * (env_instance.env.scene.timestep = 0.004125)]
    
            print("Episode : ", i)
            done = False
    
            initial_joint_state = Mesh_States[current_state_index][20-number_of_perturbable_states:] #get last 12 states, observation size is 20
            initial_joint_state = initial_joint_state + perturbation

            next_obs = env_instance.reset() #start simulation with joints edited by initial state
            next_obs = env_instance.set_joint_states_from_disturbance_profile(initial_joint_state)    
            #perturb state
    
            while not done:      # rollout single episode
                a, b, c =  policy.compute_actions([next_obs]) #get action from policy
                next_obs, reward, done, info =  env_instance.step( a.reshape(a.size,) )
                #################################################################### start Roman
                number_of_remaining_simulation_steps -= 1
                print("RLLibStates: ")
                print("Observation: ", next_obs)
                print("Observation type: ", type(next_obs))
                print("Observation Shape: ", next_obs.shape)

                #################################################################### start Roman
                number_of_remaining_simulation_steps -= 1
                print("RLLibStates: ")
                print("Observation: ", obs)
                print("Observation type: ", type(obs))
                print("Observation Shape: ", obs.shape)
                #check if we should add observation to Mesh
                if done or number_of_remaining_simulation_steps == 0: ### if episode is done (we fell) or if the episode time limit was reached 
                    if done:    ##### if we failed (walker fell before time limit was reached)
                        T_det[ (current_state_index,) + (controller_index,) + it.multi_index ] = 0 #(=failure state index)
                        #TODO: T_det[current_state_index,controller_index,disturbance_index_1,...,disturbance_index_2] = 0 #(=failure state index)
                        #pass
                    else:       ##### if we succeded (walker didnt fall before timelimit was reached)
                        
                        distance_to_closest_state_in_Mesh, index_of_closest_state_in_Mesh =  NearestNeighbors(Mesh,obs) #check if we should add state
                        if distance_to_closest_state_in_Mesh > distance_threshold: # if final state is far away enough from all other states in Mesh 
                            Mesh_States = np.concatenate( ( Mesh_States, obs[:20].reshape(1,20) ) , axis=0 ) # add state to Mesh table
                            total_number_of_states += 1   # number of states in Mesh table has increased
                            nbrs = NearestNeighbors(n_neighbors=1).fit(Mesh_States[1:,:]) # update KD-tree/ball-tree with newly added state 
                                                                                          # omitted first (failure) state
                            T_det = np.concatenate( ( T_det, np.zeros(shape=T_det.shape[1:]) ) , axis=0 ) #append T_det to include new state
                        else:                            #if we do not add new state to mesh (because new state is close to a state in existing mesh)
                            T_det[(current_state_index,) + (controller_index,) + it.multi_index ] = index_of_closest_state_in_Mesh + 1 # we add a 1 because we dont include the first failure state when finding the nearest neighoring state
                            # TODO #T_det[current_state_index,controller_index,disturbance_index_1,...,disturbance_index_2]= index_of_closest_state_in_Mesh + 1 # we add a 1 because we dont include the first failure state when finding the nearest neighoring state 
                                #pass
                    done = True # ensure that episode is done so that we dont simulate unecessary steps

                    current_state_index += 1
                    #################################################################### end Roman

####################################################################### start
#export the Mesh of States for current policy
#export the Deterministic State Transition Matrix
####################################################################### end

############################################################# create the probabllities for the disturbance profile
#probabilities = var.pdf( np.moveaxis(disturbances,0,-1) ) # make first axis the last, other axes retain their relative order # shape goes from (12,4,4,4,4,4,4,4,4,4,4,4,4) to (4,4,4,4,4,4,4,4,4,4,4,4,12) #get the probablilities for each possible disturbance of the state, based on our multivariate gaussian model probabilities are in shape of (4,4,4,4,4,4,4,4,4,4,4,4)
#probabilities.sum(axis=None) #sum of all probabilities
#probabilities = probabilities/probabilities.sum(axis=None) # normalize disturbnce probability distribution
############################################################# end create the probabllities for the disturbance profile

#############################################################create the non deterministic state transition matrix

############################################################# end create the np-det state trasistion matrix


env_instance.close()

