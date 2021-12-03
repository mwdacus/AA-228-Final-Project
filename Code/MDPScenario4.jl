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

    p_suc_drive::Beta=Beta(3,10)
    p_stop_drive::Beta=Beta(3,10)

    # termination_state::Array = [State(25, ps) for ps in -30:30]  # for t in Teams]
    termination_state::State = State(26, 0)

end

#Assign params to variable Parameters
params=ExtraParameters();

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];


𝒮=[[State(d, ps) for d=1:25 for ps=-30:30]..., params.termination_state] #for t=Teams]...,params.termination_states]

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
    # println(s)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)

    p_home_kick = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick))
    p_away_kick = p_home_kick
    p_home_two = mean.(params.p_suc_two) * (1-mean.(params.p_stop_two))
    p_home_drive = mean.(params.p_suc_drive)
    p_away_drive = mean.(params.p_suc_drive)

    # Transition to single sink terminal state
    if s.drive == 25
        prob[61*25 + 1] = 1
        return SparseCat(nextstate, prob)
    end
    # Skipping edge cases of the pointspread bound
   
    if s.pointspread <= -23
        prob[i[1] + 61] = 1
        return SparseCat(nextstate, prob)
    elseif s.pointspread >= 20
        prob[i[1] + 61] = 1
        return SparseCat(nextstate, prob)
    end

    # println("break1")
    if !iseven(s.drive) && s!=params.termination_state # prob[state + kick points - away touchdown + switch teams]
        if a==kick #&& !(s in params.termination_states)
            prob[i[1] + 61 + 1 - 6] = p_home_kick     * p_away_drive          # home make kick, away makes drive
            prob[i[1] + 61 + 1 - 6] = p_home_kick     * (1-p_away_drive)      # home make kick, away misses drive
            prob[i[1] + 61 + 1 - 6] = (1-p_home_kick) * p_away_drive    # home miss kick, away makes drive
            prob[i[1] + 61 + 1 - 6] = (1-p_home_kick) * (1-p_away_drive)  # home miss kick, away misses drive
            # prob[i[1]] = 1
            return SparseCat(nextstate,prob)
        elseif a==two #&& !(s in params.termination_states)
            prob[i[1] + 61 + 2 - 6] = p_home_two     * p_away_drive      # home make two, away makes drive
            prob[i[1] + 61 + 0 - 6] = p_home_two     * (1-p_away_drive)  # home make two, away misses drive
            prob[i[1] + 61 + 2 - 0] = (1-p_home_two) * p_away_drive      # home miss two, away makes drive
            prob[i[1] + 61 + 0 - 0] = (1-p_home_two) * (1-p_away_drive)  # home miss two, away misses drive
            # prob[i[1]] = 1
            return SparseCat(nextstate,prob)
        end
    elseif iseven(s.drive) && s!=params.termination_state # away team -- take stochastic 'kick' action, stochastic home drive
        if a==kick #&& !(s in params.termination_states)
            prob[i[1] + 61 - 1 + 6] = p_away_kick     * p_home_drive      # away make kick, home makes drive
            prob[i[1] + 61 - 0 + 6] = p_away_kick     * (1-p_home_drive)  # away make kick, home misses drive
            prob[i[1] + 61 - 1 + 0] = (1-p_away_kick) * p_home_drive      # away miss kick, home makes drive
            prob[i[1] + 61 - 0 + 0] = (1-p_away_kick) * (1-p_home_drive)  # away miss kick, home misses drive
            # prob[i[1]] = 1
            return SparseCat(nextstate,prob)
        elseif a==two # WORKAROUD: away team can't go for two
            # prob[i + 121] = 1
            prob[i[1]] =1
            return SparseCat(nextstate, prob)
        end
    else
        p[end] = 1
    end
end


#Define the termination state
termination(s::State) = s == params.termination_state #in params.termination_states

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

	solver=ValueIterationSolver(max_iterations=10)
	policy=solve(solver,mdp)

    println(policy)
# DEBUGGING
# s_test = State(1, -23)
# a_test = kick
# dist_out = T(s_test, a_test)
# println(dist_out(31 + 61 + 1 - 6))