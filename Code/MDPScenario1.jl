using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Plots

@with_kw mutable struct ExtraParameters
	#Rewards
	r_kickmade::Real=1
	r_scoremade::Real=2
	
	#Transitional Probabilities (Beta Distribution)
	p_suc_kick::Real
	p_stop_kick::Real
	p_suc_two::Real
	p_stop_two::Real
end

@enum State TD kickmade twomade missed terminalstate
@enum Action kick two
𝒜=[kick, two];
𝒮=[TD, kickmade, twomade, missed, terminalstate];


#Reward Function
function R(s::State,a::Action,s′::State)
	if s==TD && (s′==kickmade && a==kick)
		return 1
	elseif s==TD && (s′==twomade && a==two)
		return 2
	else
		return 0
	end
end

#Transition Model
function T(s::State,a::Action)
	if a==kick && s==TD
		p=(params.p_suc_kick)*(1-params.p_stop_kick)
		return SparseCat([TD, kickmade, twomade, missed, terminalstate], 
			[0 p 0 1-p 0])
	elseif a==two && s==TD
		p=(params.p_suc_two)*(1-params.p_stop_two)
		return SparseCat([TD, kickmade, twomade, missed, terminalstate],[0 0 p 1-p 0])
	else
		return SparseCat([TD, kickmade, twomade, missed, terminalstate], [0 0 0 0 1])
	end
end	

#Termination State
termination(s::State)=s==𝒮[5]

#Discount Factor
γ=0.9

abstract type FieldGoal <: MDP{State, Action} end

#MDP Definition
mdp = QuickMDP(FieldGoal,
	states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = 𝒮,
    isterminal   = termination);


p_home_kick=range(0,1,length=1000)
p_opp_kick=range(0.2,0.8,length=4)
p_home_two=range(0,1,length=1000)
p_opp_two=range(0.2,0.8,length=4)
v=zeros(length(p_opp_kick),length(p_home_kick))

for j in 1:length(p_opp_kick)
	for i in 1:length(p_home_kick)
		global params
		params=ExtraParameters(p_suc_kick=p_home_kick[i],p_stop_kick=p_opp_kick[j],p_suc_two=0,p_stop_two=0)
		solver=ValueIterationSolver(max_iterations=50)
		policy=solve(solver,mdp)
		v[j,i]=value(policy,TD)
	end
end
    
for i in 1:length(p_opp_kick)
	plot!(p_home_kick,v[i,:])
end
