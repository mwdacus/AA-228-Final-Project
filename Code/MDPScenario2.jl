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
using TabularTDLearning
using POMDPPolicies

struct State
	PS::Int
end

#Define the State Space
#Enter Parameters
@with_kw mutable struct ExtraParameters
	#Rewards
	r_kickmade::Real=1
	r_scoremade::Real=2
	
	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick::Beta=Beta(9,1)
	p_stop_kick::Beta=Beta(1,9)
	p_suc_two::Beta=Beta(6,4)
	p_stop_two::Beta=Beta(5,5)

	#Define the win state, lose state, and termination state
	win_state=State(100)
	lose_state=State(101)
	termination_state=State(102)
end

#Assign params to variable Parameters
params=ExtraParameters();

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(x) for x=-70:70]...,params.win_state,
	params.lose_state,params.termination_state]

#Only dependent on state
function R(s::State,a::Action)
	return (s==params.win_state ? 10 : 0) +
	 (s==params.lose_state ? -10 : 0) + 
	 (a==kick && s.PS>0 ? 1*exp(-s.PS/15) : 0) +
	 (a==kick && gcd(s.PS+1,7)==7 && s.PS<0 ? -2 : 0) + 
	 (a==kick && s.PS<0 ? 1*exp(s.PS/15) : 0) +
	 (a==two && gcd(s.PS+1,7)==7 && s.PS<0 ? 2*exp(s.PS/15) : 0) +
	 (a==two && gcd(s.PS+1,7)==7 && s.PS>0 ? 2*exp(-s.PS/15) : 0)
end


function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)
	if a==kick && s!=params.win_state && s!=params.lose_state && s!=params.termination_state
		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
		prob[i[1]]=1-prob[i[1]+1]
		return SparseCat(nextstate,prob)
	elseif s!=params.win_state && s!=params.lose_state && s!=params.termination_state
		prob[i[1]+1]=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick))
		prob[i[1]]=1-prob[i[1]+1]
		return SparseCat(nextstate,prob)
	else
		prob[end]=1
		return SparseCat(nextstate,prob)
	end
end

#Define the termination state
termination(s::State)=s==params.termination_state

#Define Discount Factor
γ=0.9;

#Define the mdp
abstract type FieldGoal <: MDP{State, Action} end

mdp = QuickMDP(FieldGoal,
	states       = 𝒮,
	actions      = 𝒜,
	transition   = T,
	reward       = R,
	discount     = γ,
	initialstate = 𝒮,
	isterminal   = termination);

#solve using Value Iteration
solver=ValueIterationSolver(max_iterations=100)
VI_policy=solve(solver,mdp)

#solve using Q-Learning (with an Exploration policy)
q_mdp = QuickMDP(FieldGoal,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ, # custom discount for visualization of Q-learning policy
    initialstate = 𝒮,
    isterminal   = termination);

q_learning_solver = QLearningSolver(n_episodes=50,
	learning_rate=0.3,
	exploration_policy=EpsGreedyPolicy(q_mdp, 0.5),
	verbose=false);
q_learning_policy = solve(q_learning_solver, q_mdp);

