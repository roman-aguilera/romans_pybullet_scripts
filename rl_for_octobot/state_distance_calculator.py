import cvxpy as cp
import numpy as np
'''
state_size = 30 
num_non_failure_states = 20
threshold = 10

C = np.random.randn(state_size, num_non_failure_states) #allstates
c = np.random.randn(state_size) #new state
C = np.concatenate(   ( np.ones( (state_size,1) )*np.inf, C  ) , axis=1   ) #adds failure state at beginning to create ful state table

print("B Matrix : ", C)
print("B shape :", C.shape)

newStateCandidate, addStateIndicator = is_distance_threshold_exceeded(C, c, threshold)

if addStateIndicator:
	B = np.concatenate( (B, newStateCandidate  ) , axis=1   )

print("B Matrix : ", B)
print("B shape :", B.shape)
'''
def is_distance_threshold_exceeded(allStates, newState, distanceThreshold):
	
	m = allStates[0] #size of observation states (row length)
	n = allStates.shape[1] - 1 #number of states in state table (column length) 
				   #exclude fist state, i.e. failure state, because it is denoted by infinity and cvpy doesnt like that
	
	A = allStates[:,1:] #dont account for first column (failure state) 
	b = newState


	x = cp.Variable(n)
	objective = cp.Minimize(cp.sum_squares(A@x - b))
	#constraints = [0 <= x, x <= 1]
	prob = cp.Problem(objective) #, constraints)

	# The optimal objective value is returned by `prob.solve()`.
	result = prob.solve()
#	print("Minimum distance : ",result)
	# The optimal value for x is stored in `x.value`.
#	print("Weights: ", x.value)
	# The optimal Lagrange multiplier for a constraint is stored in
	# `constraint.dual_value`.
	#print(constraints[0].dual_value)

	newStateReshaped = np.expand_dims(newState,1) #states of shape (m,) need to be of shape (m,1)
	
	if result > distanceThreshold:
		addState = True
	else:
		addState = False
	
	return newStateReshaped , addState


state_size = 30
num_non_failure_states = 20
threshold = 10

C = np.random.randn(state_size, num_non_failure_states) #allstates
c = np.random.randn(state_size) #new state
C = np.concatenate(   ( np.ones( (state_size,1) )*np.inf, C  ) , axis=1   ) #adds failure state at beginning to create ful state table

#print("C Matrix : ", C)
print("C shape :", C.shape)

newStateCandidate, addStateIndicator = is_distance_threshold_exceeded(C, c, threshold)

print("State was added : ", addStateIndicator)

if addStateIndicator:
        C = np.concatenate( (C, newStateCandidate  ) , axis=1   )

#print("C Matrix : ", C)
print("C shape :", C.shape)


