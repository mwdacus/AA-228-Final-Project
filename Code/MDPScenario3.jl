using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions

#Define the State Space
struct State
	x::Int
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

	win_state=State(41)
	lose_state=State(42)
	termination_state=State(43)
end

#Assign params to variable Parameters
params=ExtraParameters();

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(x) for x=-40:40]...,params.win_state,
	params.lose_state,params.termination_state]

#Only dependent on state
function R(s::State,a::Action,s′::State)
    if s′.x-s.x==1 && a==kick
		return 1/40*(s.x)
    elseif s′.x-s.x==2 && a==two && s.x>0
		return 1/20*s.x
	elseif s′.x-s.x==2 && a==two && s.x<0
		return -1/20*s.x
	else
		return 0
    end
end	

function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)
	if a==kick && (s!=State(41) && s!=State(42) && s!=State(43))
		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
		prob[i[1]]=1-prob[i[1]+1]
		return SparseCat(nextstate,prob)
	elseif a==kick && (s!=State(41) && s!=State(42) && s!=State(43))
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
	