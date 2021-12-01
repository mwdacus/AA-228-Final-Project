using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions

#Define the State Space
mutable struct State
	drive::Int
	pointspread::Int
end

#Enter Parameters
@with_kw mutable struct ExtraParameters
	# Random problem Parametersexit()
	num_drives::Int = 12
	pointspread_max::Int = 60
	pointspread_vector::Array = collect(-pointspread_max:pointspread_max)
	pointspread_size::Int = size(pointspread_vector)[1]

	#Rewards
	# r_kickmade::Real=1
	# r_scoremade::Real=2
	r_vector::Array = zeros(pointspread_size)
	# r_vector[1] = 1

	
	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick::Beta=Beta(9,1)
	p_stop_kick::Beta=Beta(1,9)
	p_suc_two::Beta=Beta(2,8)
	p_stop_two::Beta=Beta(7,3)

	termination_states = [State(num_drives,i) for i in -pointspread_max:pointspread_max]
end

#Assign params to variable Parameters
params=ExtraParameters();

#Workaround for bug: updating r_vector inside of ExtraParameters won't work
params.r_vector[1:params.pointspread_max-2 ] .= -1
params.r_vector[ params.pointspread_max+2:params.pointspread_size] .= 1
r_closegame = [-100, 20, 100] 
params.r_vector[params.pointspread_max-1:params.pointspread_max+1] .= r_closegame

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];

#2D State Space , [12 drives + termainal state, 81 pointspreads]
𝒮=[[State(d, ps) for d=-params.pointspread_max:params.pointspread_max for ps=1:params.num_drives+1]...]

#Only dependent on state
function R(s::State,a::Action,s′::State)
	pointspread_final = s′.pointspread
	return params.r_vector[pointspread_final + params.pointspread_max]
end	

#Rollout a game from the current state. 'result' represents the outcome of the kick/two attempt post-TD ~ [0,1,2].
function SimulateSubgame(s::State, extra_points::Int)
	s′::State = State(s.drive + 1, s.pointspread + extra_points)
	p_home_touchdown = mean.(params.p_suc_two)
	p_opp_touchdown  = mean.(params.p_stop_two)
	dist_home = Bernoulli(p_home_touchdown)
	dist_opp  = Bernoulli(p_opp_touchdown)
	samples_home = rand(dist_home, params.num_drives+1 - s′.drive)  #Draw drive results for the remaining drives
	samples_opp  = rand(dist_opp , params.num_drives+1 - s′.drive)
	pointspread_final = s′.pointspread + 7*sum(samples_home) - 7*sum(samples_opp)
	s′.pointspread = pointspread_final
	return s′
end

# function T(s::State,a::Action)
# 	nextstate=𝒮
# 	prob=zeros(length(nextstate))
# 	i=findall(x->x==s,𝒮)
# 	if a==kick && (s!=State(41) && s!=State(42) && s!=State(43))
# 		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
# 		prob[i[1]]=1-prob[i[1]+1]
# 		return SparseCat(nextstate,prob)
# 	elseif a==kick && (s!=State(41) && s!=State(42) && s!=State(43))
# 		prob[i[1]+2]=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two))
# 		prob[i[1]]=1-prob[i[1]+2]
# 		return SparseCat(nextstate,prob)
#     else
# 		prob[end]=1
# 		return SparseCat(nextstate,prob)
# 	end
# end


# #Define the termination state
# termination(s::State)=s==params.termination_state

# #Define Discount Factor
# γ=0.9

# abstract type FieldGoal <: MDP{State, Action} end

# mdp = QuickMDP(FieldGoal,
# 	states       = 𝒮,
#     actions      = 𝒜,
#     transition   = T,
#     reward       = R,
#     discount     = γ,
#     initialstate = 𝒮,
#     isterminal   = termination);

#     solver=ValueIterationSolver(max_iterations=1000)
#     policy=solve(solver,mdp)
	

# DEBUGGING PRINTOUTS
start_state = State(3, -1)
test_result = 2
test_action = kick
final_state = SimulateSubgame(start_state, test_result)
# final_state.pointspread = 1
test_reward = R(start_state, test_action, final_state)

print("Start state: ")
println(start_state)
print("Game end state: ")
println(final_state)
print("Reward: ")
println(test_reward)