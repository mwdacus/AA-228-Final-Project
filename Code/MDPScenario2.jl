#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Use Value Iteration to Determine the best policy after scoring a touchdown 
#(either kick or go for two point conversion) based on point spread
#Input: 
    #Rewards
    #Extra point completion percentage (p_suc_kick), 
    #Opponent extra point stopping percentage (p_stop_kick),
    #two point conversion completion percentage (p_suc_two)
    #Opponent two point stopping percentage (p_stop_two)
#Output: policy (action for State)
#----------------------------------------------------------------------------------

using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions

#Define the State Space
struct State
	PS::Int
end

#Enter Parameters
@with_kw mutable struct ExtraParameters
	#Rewards
	r_kickmade::Real=1
	r_scoremade::Real=2
	
	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick::Beta=Beta(9,1)
	p_stop_kick::Beta=Beta(1,9)
	p_suc_two::Beta=Beta(2,8)
	p_stop_two::Beta=Beta(7,3)

	win_state=State(31)
	lose_state=State(32)
	termination_state=State(33)
end

#Assign params to variable Parameters
params=ExtraParameters();

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(x) for x=-30:30]...,params.win_state,
	params.lose_state,params.termination_state]

#Only dependent on state
function R(s::State,a::Action,s′::State)
    
end	

function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)
	if a==kick && (s!=State(31) && s!=State(32) && s!=State(33))
		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
		prob[i[1]]=1-prob[i[1]+1]
		return SparseCat(nextstate,prob)
	elseif a==kick && (s!=State(31) && s!=State(32) && s!=State(33))
		prob[i[1]+2]=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two))
		prob[i[1]]=1-prob[i[1]+2]
		return SparseCat(nextstate,prob)
    else
		prob[end]=1
		return SparseCat(nextstate,prob)
	end
end


#Define the termination state
termination(s::State)=s==params.termination_state

#Define Discount Factor
γ=0.9

abstract type FieldGoal <: MDP{State, Action} end

mdp = QuickMDP(FieldGoal,
	states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = 𝒮,
    isterminal   = termination);

    solver=ValueIterationSolver(max_iterations=1000)
    policy=solve(solver,mdp)
	