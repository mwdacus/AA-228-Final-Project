using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Distributions
using LinearAlgebra
using CSV, Tables # Writing to CSV files
using DataFrames  # Not sure if this will end up being necessary

#Define the State Space
mutable struct State
	drive::Int
	pointspread::Int
end

#Enter Parameters
@with_kw mutable struct ExtraParameters
	# Random problem Parametersexit()
	num_drives::Int = 12
	pointspread_max::Int = 10
	pointspread_vector::Array = collect(-pointspread_max:pointspread_max)
	pointspread_size::Int = size(pointspread_vector)[1]

	#Rewards
	r_vector::Array = zeros(pointspread_size)

	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick  = 0.1
	p_stop_kick = 0.1
	p_suc_two   = 0.1
	p_stop_two  = 0.1
	p_home_kick = p_suc_kick * (1-p_stop_kick)  # unnormalized
	p_opp_kick  = (1-p_suc_kick) * p_suc_kick
	p_home_two  = p_suc_two * (1-p_stop_two)
	p_opp_two   = (1-p_suc_two) * p_stop_two
	dist_home_kick = Bernoulli(p_home_kick / sum([p_home_kick, p_opp_kick]))  # normalized distributions to draw from
	dist_opp_kick  = Bernoulli(p_opp_kick  / sum([p_home_kick, p_opp_kick]))
	dist_home_two  = Bernoulli(p_home_two  / sum([p_home_two, p_opp_two]))
	dist_opp_two   = Bernoulli(p_opp_two   / sum([p_home_two, p_opp_two]))

	termination_states = [State(num_drives,i) for i in -pointspread_max:pointspread_max]
end

#Assign params to variable Parameters
params=ExtraParameters();

#Workaround for bug: updating r_vector inside of ExtraParameters won't work
params.r_vector[1:params.pointspread_max-1] .= collect(-params.pointspread_max:-2)
params.r_vector[params.pointspread_max+3:params.pointspread_size] .= collect(2 : params.pointspread_max)
r_closegame = [-100, 20, 100]
params.r_vector[params.pointspread_max:params.pointspread_max+2] .= r_closegame

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];

#2D State Space , [12 drives + termainal state, 81 pointspreads]
𝒮=[[State(d, ps) for d=1:params.num_drives for ps=-params.pointspread_max:params.pointspread_max]...]
# 𝒮=[[State(d, ps) for d=-params.pointspread_max:params.pointspread_max for ps=1:params.num_drives+1]...]  #Includes terminal states...not sure if useful yet

#Only dependent on state
# function R(s::State,a::Action,s′::State)
function R(s′::State)
	pointspread_final = s′.pointspread
	if pointspread_final < -params.pointspread_max
		pointspread_final = -params.pointspread_max
	elseif pointspread_final > params.pointspread_max
		pointspread_final = params.pointspread_max
	end
	return params.r_vector[pointspread_final + params.pointspread_max + 1]
end

# Performs the attempt for kick/two, returns the probabilistic outcome
function doAttempt(s::State, a::Action)
	if a == kick
		return 1 * rand(params.dist_home_kick, 1)[1]   # Comes in vector form, indexing to extract
	elseif a == two
		return 2 * rand(params.dist_home_two, 1)[1]
	end
end

# Rollout a game from the current state. 'result' represents the outcome of the kick/two attempt post-TD ~ [0,1,2].
function SimulateSubgame(s::State, extra_points::Int)
	s′::State = State(s.drive + 1, s.pointspread + extra_points)
	dist_TD_home = params.dist_home_two
	dist_TD_opp  = params.dist_opp_two
	samples_home = rand(dist_TD_home, params.num_drives+1 - s′.drive)  #Draw drive results for the remaining drives
	samples_opp  = rand(dist_TD_opp , params.num_drives+1 - s′.drive)
	pointspread_final = s′.pointspread + 7*sum(samples_home) - 7*sum(samples_opp)
	s′.pointspread = pointspread_final
	return s′
end

# Runs n iterations of a full game for each state-action pair. Returns average reward for each state-action pair
function SolveGames(num_iters::Int)
	Q::Array = zeros(size(𝒮)[1], size(𝒜)[1])
	i=1
	for s in 𝒮
		j=1
		for a in 𝒜
			total_reward = 0
			for iter in 1:num_iters
				points = doAttempt(s, a)
				total_reward += R(SimulateSubgame(s, points))
				# Update the Q function for the given state-action pair with the average end-game reward
				Q[i,j] = total_reward / num_iters
			end
			j += 1
		end
		i += 1
	end
	return Q
end

# Returns the optimal policy from each state given the state-action pair rewards from SolveGames()
function GetPolicy(Q::Array)
	Pi = zeros(size(𝒮)[1])
	for i in 1:size(Q)[1]
		if Q[i,1] >= Q[i,2]
			Pi[i] = 1
		else
			Pi[i] = 2
		end
	end
	Pi_reshaped = reshape(Pi, (params.num_drives, params.pointspread_size))
	return Pi_reshaped
end

function readCSV(infile::String)
	df = DataFrame(CSV.File(infile))
    D = Matrix(df)
	println(D)
end

function computePolicy(n::Int)
	Q = SolveGames(n)
	Pi = GetPolicy(Q)
	# CSV.write("Data/Scenario3_Policies/test_matchup.csv",  Tables.table(test_Pi), writeheader=false)
end
# DEBUGGING / TESTING

params.p_suc_kick  = 0.9
params.p_stop_kick = 0.1
params.p_suc_two   = 0.2
params.p_stop_two  = 0.7

test_state = State(3,-1)
test_action = kick
test_Q = SolveGames(1000)
test_Pi = GetPolicy(test_Q)
CSV.write("Data/Scenario3_Policies/test_matchup.csv",  Tables.table(test_Pi), writeheader=false)

