#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Use Value Iteration to Determine the best policy after scoring a field goal (either kick or go for two point conversion)
#Input: 
    #Rewards
    #Extra point completion percentage (p_suc_kick), 
    #Opponent extra point stopping percentage (p_stop_kick),
    #two point conversion completion percentage (p_suc_two)
    #Opponent two point stopping percentage (p_stop_two)
#Output: policy (action for State TD,kickmade,twomade,missed and terminalstate)
#----------------------------------------------------------------------------------

using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration

#Define Parameters
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

#Assign Parameters to variable Parameters
params=ExtraParameters(p_suc_kick=0.8,p_stop_kick=0.2,p_suc_two=0.4,p_stop_two=0.6);

#Define State and Action Space
@enum State TD kickmade twomade missed terminalstate
@enum Action kick two
𝒜=[kick, two];
𝒮=[TD, kickmade, twomade, missed, terminalstate];

#Define Reward Function
function R(s::State,a::Action,s′::State)
	if s==TD && (s′==kickmade && a==kick)
		return 1
	elseif s==TD && (s′==twomade && a==two)
		return 2
	else
		return 0
	end
end

#Define Transition Function
function T(s::State,a::Action)
	if a==kick && s==TD
		p=(params.p_suc_kick)*(1-(params.p_stop_kick))
		return SparseCat([TD, kickmade, twomade, missed, terminalstate], 
			[0 p 0 1-p 0])
	elseif a==two && s==TD
		p=(params.p_suc_two)*(1-(params.p_stop_two))
		return SparseCat([TD, kickmade, twomade, missed, terminalstate],[0 0 p 1-p 0])
	else
		return SparseCat([TD, kickmade, twomade, missed, terminalstate], [0 0 0 0 1])
	end
end	

#Define Termination State
termination(s::State)=s==𝒮[5]

#Define Discount Factor
γ=0.9

#Define MDP
abstract type FieldGoal <: MDP{State, Action} end
mdp = QuickMDP(FieldGoal,
	states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = 𝒮,
    isterminal   = termination);


#Solve for Best Policy using Value Iteration
solver=ValueIterationSolver(max_iterations=50)
policy=solve(solver,mdp)