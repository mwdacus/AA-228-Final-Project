using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions

#Define the State Space
struct State
	drive::Int
	point_spread::Int
end

#Enter Parameters
@with_kw mutable struct ExtraParameters
	# Random problem Parameters
	pointspread_max::Int = 40
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

	termination_states = [State(13,i) for i in -40:40]
end

#Assign params to variable Parameters
params=ExtraParameters();

params.r_vector[1:params.pointspread_max-2 ] .= -1
params.r_vector[ params.pointspread_max+2:params.pointspread_size] .= 1
r_closegame = [-100, 20, 100] 
params.r_vector[params.pointspread_max-1:params.pointspread_max+1] .= r_closegame

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(d, ps) for d=-40:40 for ps=1:13]...]

#Only dependent on state
function R(s::State,a::Action,s′::State)
	pointspread_final = s'.point_spread
	return params.r_vector[pointspread_final + pointspread_max]
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
# println(termination_states)
println(params.r_vector)
