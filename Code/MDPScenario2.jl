using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions

@with_kw mutable struct ExtraParameters
	#Rewards
	r_kickmade::Real=1
	r_scoremade::Real=2
	
	#Transitional Probabilities (Beta Distribution)
	p_suc_kick::Beta=Beta(9,1)
	p_stop_kick::Beta=Beta(1,9)
	p_suc_two::Beta=Beta(2,8)
	p_stop_two::Beta=Beta(7,3)

	win_state=State(201)
	lose_state=State(202)
	termination_state=State(203)
end

params=ExtraParameters();

abstract type FieldGoal <: MDP{State, Action} end


@enum Action kick two
𝒜=[kick, two];

struct State
	x::Int
end

𝒮=[[State(x) for x=-200:200]...,params.win_state,
	params.lose_state,params.termination_state]

termination(s::State)=s==params.termination_state

γ=0.9

#Only dependent on state
function R(s::State,a::Action,s′::State)
    if s==State(201)
        return 0
    elseif s==State(202)
        return 0
    elseif s==State(s′.x-1) && s.x>0
        return 1
    elseif s==State(s′.x-2) && s.x>0
        return 2
    elseif s==State(s′.x-1) && (s.x<-125 && s.x>=-150) && a==kick
        return -2
    elseif s==State(s′.x-1) && (s.x<-160 && s.x>=-200) && a==kick
        return -4
    #elseif s==State(s′.x-2) && (s.x<-125 && s.x>=-150) && a==two
    #	return 2
    else 
        return 0	
    end
end		

function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)
	if a==kick && (s!=State(201) && s!=State(202) && s!=State(203))
		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
		prob[i[1]]=1-prob[i[1]+1]
		return SparseCat(nextstate,prob)
	elseif a==kick && (s!=State(201) && s!=State(202) && s!=State(203))
		prob[i[1]+2]=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two))
		prob[i[1]]=1-prob[i[1]+2]
		return SparseCat(nextstate,prob)
    else
		prob[end]=1
		return SparseCat(nextstate,prob)
	end
end

termination(s::State)=s==𝒮[5]

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