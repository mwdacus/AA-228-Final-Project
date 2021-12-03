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

# @enum Team home away
# Teams = [home,away]

struct State
	drive::Int
	pointspread::Int
    # team::Team   # 0=home, 1=away
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
	p_suc_two::Beta=Beta(2,8)
	p_stop_two::Beta=Beta(7,3)

    p_suc_drive::Beta=Beta(5,5)
    p_stop_drive::Beta=Beta(5,5)

    termination_states::Array = [State(25, ps) for ps in -30:30]  # for t in Teams]
end

#Assign params to variable Parameters
params=ExtraParameters();

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(d, ps) for d=1:24 for ps=-30:30]..., params.termination_states] #for t=Teams]...,params.termination_states]

#Only dependent on state
function R(s::State,a::Action,s′::State)
    if s′.drive == 25  # End game rewards
        if s′ == -1
            return -1000
        elseif s′ == 0
            return 200
        elseif s′ == 1
            return 1000
        else
            return s′.pointspread
        end
    else  # Immediate rewards for kick/two
        if s′.pointspread-s.pointspread==1 && a==kick
            return 1
        elseif s′.pointspread-s.pointspread==2 && a==two
            return 2
        else
            return 0	
        end
    end
end	


function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)

    if !iseven(s.drive)  # prob[state + kick points - away touchdown + switch teams]
        if a==kick && !(s in params.termination_states)
            prob[i[1] + 61 + 1 - 6] = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick)) * mean.(params.p_suc_drive)    # make kick, away makes drive
            prob[i[1] + 61 + 0 - 6] = 1-prob[i[1] + 61 + 1 - 6] * mean.(params.p_suc_drive)                             # make kick, away misses drive
            prob[i[1] + 61 + 1 - 0] = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick)) * mean.(params.p_stop_drive)   # miss kick, away makes drive
            prob[i[1] + 61 + 0 - 0] = 1-prob[i[1] + 61 + 1 - 0] * mean.(params.p_suc_drive)                            # miss kick, away misses drive
            return SparseCat(nextstate,prob)

        elseif a==two && !(s in params.termination_states)
            prob[i[1] + 61 + 2 - 6] = mean.(params.p_suc_two) * (1-mean.(params.p_stop_two)) * mean.(params.p_suc_drive)    # make kick, away makes drive
            prob[i[1] + 61 + 0 - 6] = 1-prob[i[1] + 61 + 2 - 6] * mean.(params.p_suc_drive)                             # make kick, away misses drive
            prob[i[1] + 61 + 2 - 0] = mean.(params.p_suc_two) * (1-mean.(params.p_stop_two)) * mean.(params.p_stop_drive)   # miss kick, away makes drive
            prob[i[1] + 61 + 0 - 0] = 1-prob[i[1] + 61 + 2 - 0] * mean.(params.p_suc_drive)                            # miss kick, away misses drive
        else
            prob[end]=1
            return SparseCat(nextstate,prob)
        end

    else  # away team -- take stochastic 'kick' action, stochastic home drive
        if a==kick && !(s in params.termination_states)
            prob[i[1] + 61 - 1 + 6] = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick)) * mean.(params.p_suc_drive)    # make kick, away makes drive
            prob[i[1] + 61 - 0 + 6] = 1-prob[i[1] + 61 - 1 + 6] * mean.(params.p_suc_drive)                             # make kick, away misses drive
            prob[i[1] + 61 - 1 + 0] = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick)) * mean.(params.p_stop_drive)   # miss kick, away makes drive
            prob[i[1] + 61 - 0 + 0] = 1-prob[i[1] + 61 - 1 + 0] * mean.(params.p_suc_drive)                            # miss kick, away misses drive
            return SparseCat(nextstate,prob)
        else # WORKAROUD: away team can't go for two
            prob[i + 121] = 1
            return SparseCat(nextstate, prob)
        end
    end
end


#Define the termination state
termination(s::State) = s in params.termination_states

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
	# policy=solve(solver,mdp)


# DEBUGGING

s_test = State(1, 5, home)
a_test = kick
T(s_test, a_test)