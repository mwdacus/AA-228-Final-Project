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
using TickTock

# Define the State Space
struct State
	drive::Int
	pointspread::Int
end

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];

#Enter Parameters
@with_kw mutable struct ExtraParameters
	#Rewards
	r_kickmade::Real = 1 
	r_scoremade::Real = 2

    #State Space parameters
    num_drives = 4  # Keep it even!!! each team get n/2 drives
    pointspread_max::Int = (num_drives/2) * 8
    pointspread_range::Int = pointspread_max*2 + 1
	
	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick::Beta  = Beta(9,1)
	p_stop_kick::Beta = Beta(1,9)
	p_suc_two::Beta   = Beta(9,1)
	p_stop_two::Beta  = Beta(1,9)

    p_suc_drive::Beta  = Beta(3,7)
    p_stop_drive::Beta = Beta(3,7)

    termination_state::State = State(num_drives + 2, 0)
    win_state::State  = State(num_drives + 2, 1)
    lose_state::State = State(num_drives + 2,-1)

end

#Assign params to variable Parameters
params=ExtraParameters();

𝒮=[[State(d, ps) for d=1:params.num_drives+1 for ps=-params.pointspread_max:params.pointspread_max]..., 
    params.win_state, params.lose_state, params.termination_state]

function R(s::State,a::Action)
	return (s==params.win_state ? 1000 : 0) +
	       (s==params.lose_state ? -1000 : 0) + 
	       (a==kick && s.pointspread>0 ? 1*exp(-s.pointspread/15) : 0) +
	       (a==kick && gcd(s.pointspread+1,7)==7 && s.pointspread<0 ? -2 : 0) + 
	       (a==kick && s.pointspread<0 ? 1*exp(s.pointspread/15) : 0) +
	       (a==two && gcd(s.pointspread+1,7)==7 && s.pointspread<0 ? 2*exp(s.pointspread/15) : 0) +
	       (a==two && gcd(s.pointspread+1,7)==7 && s.pointspread>0 ? 2*exp(-s.pointspread/15) : 0)
end

function T(s::State,a::Action)
	nextstate=𝒮
	prob=zeros(length(nextstate))
	i=findall(x->x==s,𝒮)

    one_drive = params.pointspread_range

    p_home_kick = mean.(params.p_suc_kick) * (1-mean.(params.p_stop_kick))
    p_away_kick = p_home_kick
    p_home_two = mean.(params.p_suc_two) * (1-mean.(params.p_stop_two))
    p_home_drive = mean.(params.p_suc_drive)
    p_away_drive = mean.(params.p_suc_drive)

    # Transition to single sink terminal state
    if s.drive == params.num_drives+1  # Game over, take pointspread and transition to win/lose
        if s.pointspread >= 0  # If winning/tied in Drive 25, transition to win state
            prob[end - 2] = 1
        elseif s.pointspread < 0
            prob[end-1] = 1  #
        # else
        #     prob[end] = 1
        end
        return SparseCat(nextstate, prob)
    elseif s.drive == params.num_drives+2  # In win/lose state, transition to terminal state
        prob[end] = 1
        return SparseCat(nextstate, prob)
    end

    # Never violate state space (HOKAJ: resolved by next condition?)
    if s.pointspread < -params.pointspread_max + 8 || s.pointspread > params.pointspread_max - 8
        prob[end] = 1
        return SparseCat(nextstate, prob)
    # Limiting transition space to possible states
    elseif s.pointspread < -8*s.drive || s.pointspread > 8*s.drive
        prob[end] = 1
        return SparseCat(nextstate, prob)
    end

    # Home just made TD, goes for kick/two, then away goes for a drive
    if !iseven(s.drive) && s!=params.termination_state # prob[state + kick points - away touchdown + switch teams]
        if a==kick
            prob[i[1] + one_drive + 1 - 6] = p_home_kick     * p_away_drive          # home make kick, away makes drive
            prob[i[1] + one_drive + 1 - 0] = p_home_kick     * (1-p_away_drive)      # home make kick, away misses drive
            prob[i[1] + one_drive + 0 - 6] = (1-p_home_kick) * p_away_drive          # home miss kick, away makes drive
            prob[i[1] + one_drive + 0 - 0] = (1-p_home_kick) * (1-p_away_drive)      # home miss kick, away misses drive
            return SparseCat(nextstate,prob)
        elseif a==two
            prob[i[1] + one_drive + 2 - 6] = p_home_two     * p_away_drive      # home make two, away makes drive
            prob[i[1] + one_drive + 2 - 0] = p_home_two     * (1-p_away_drive)  # home make two, away misses drive
            prob[i[1] + one_drive + 0 - 6] = (1-p_home_two) * p_away_drive      # home miss two, away makes drive
            prob[i[1] + one_drive + 0 - 0] = (1-p_home_two) * (1-p_away_drive)  # home miss two, away misses drive
            return SparseCat(nextstate,prob)
        end
    
    elseif s.drive == params.num_drives
        if a==kick
            prob[i[1] + one_drive - 1] = p_away_kick  #   * p_home_drive      # away make kick, home makes drive
            # prob[i[1] + one_drive - 1] = p_away_kick     * (1-p_home_drive)  # away make kick, home misses drive
            prob[i[1] + one_drive - 0] = (1-p_away_kick)  #* p_home_drive      # away miss kick, home makes drive
            # prob[i[1] + one_drive - 0] = (1-p_away_kick) * (1-p_home_drive)  # away miss kick, home misses drive
            return SparseCat(nextstate,prob)
        elseif a==two  # Opponent can't go for two
            prob[end] = 1
            return SparseCat(nextstate, prob)
        end
    
    
        # Away just made TD, takes action, then home goes for a drive
    elseif iseven(s.drive) && s!=params.termination_state # away team -- take stochastic 'kick' action, stochastic home drive
        if a==kick
            prob[i[1] + one_drive - 1 + 6] = p_away_kick     * p_home_drive      # away make kick, home makes drive
            prob[i[1] + one_drive - 1 + 0] = p_away_kick     * (1-p_home_drive)  # away make kick, home misses drive
            prob[i[1] + one_drive - 0 + 6] = (1-p_away_kick) * p_home_drive      # away miss kick, home makes drive
            prob[i[1] + one_drive - 0 + 0] = (1-p_away_kick) * (1-p_home_drive)  # away miss kick, home misses drive
            return SparseCat(nextstate,prob)
        elseif a==two  # Opponent can't go for two
            prob[end] = 1
            return SparseCat(nextstate, prob)
        end
    # Home can't score touchdown after last drive

    else
        # HOKAJ - redundant, delete later
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
println("Computing Value Iteration")
tick()
VI_policy=solve(solver,mdp)
tock()

q_mdp = QuickMDP(FieldGoal,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ, # custom discount for visualization of Q-learning policy
    initialstate = 𝒮,
    isterminal   = termination);

q_learning_solver = QLearningSolver(n_episodes=100,
	learning_rate=0.3,
	exploration_policy=EpsGreedyPolicy(q_mdp, 0.5),
	verbose=false);
println("Computing Q-Learning Policy...")
tick()
# q_learning_policy = solve(q_learning_solver, q_mdp);
tock()


######################
# state_test = State(1,0)
# action_test = kick
# T_test = T(state_test, action_test)