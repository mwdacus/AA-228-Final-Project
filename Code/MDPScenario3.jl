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

	#Transitional Probabilities (Beta Distribution, Default Values)
	p_suc_kick  = 0.0  # dummy values, actually assigned later using CSV data
	p_stop_kick = 0.0
	p_suc_two   = 0.0
	p_stop_two  = 0.0

	termination_states = [State(num_drives,i) for i in -pointspread_max:pointspread_max]
end

#Assign params to variable Parameters
params=ExtraParameters();

# Define 2nd-order parameters
mutable struct SecondParameters
	p_home_kick::Float64
	# p_opp_kick::Float64
	p_home_two::Float64
	p_opp_two::Float64
	dist_home_kick::Bernoulli
	# dist_opp_kick::Bernoulli
	dist_home_two::Bernoulli
	dist_opp_two::Bernoulli

	pointspread_vector::Array
	pointspread_size::Int
	r_vector::Array

end

# Secondary Parameter initalization
params_2 = SecondParameters(0,0,0,
							bern,bern,bern,
							[], 0, []);

# Update 2nd-order parameters that are dependent on the ExtraParameters(). Necessary when probabilities are updated mid-game or outside struct
function updateSecondParams(input_param::SecondParameters)
	input_param.p_home_kick = params.p_suc_kick * (1-params.p_stop_kick)  # unnormalized, probability of home team making the kick
	# p_opp_kick  = (1-params.p_suc_kick) * params.p_suc_kick   # you fail, 
	input_param.p_home_two  = params.p_suc_two * (1-params.p_stop_two)
	input_param.p_opp_two   = (1-params.p_suc_two) * params.p_stop_two

	input_param.dist_home_kick = Bernoulli(input_param.p_home_kick) # / sum([p_home_kick, p_opp_kick]))  # normalized distributions to draw from
	# dist_opp_kick  = Bernoulli(p_opp_kick ) # / sum([p_home_kick, p_opp_kick]))
	input_param.dist_home_two  = Bernoulli(input_param.p_home_two ) # / sum([p_home_two, p_opp_two]))
	input_param.dist_opp_two   = Bernoulli(input_param.p_opp_two  ) # / sum([p_home_two, p_opp_two]))

	input_param.pointspread_vector = collect(-params.pointspread_max:params.pointspread_max)
	input_param.pointspread_size = size(params.pointspread_vector)[1]
	input_param.r_vector = zeros(params.pointspread_size)

	#Workaround for bug: updating r_vector inside of ExtraParameters won't work
	input_param.r_vector[1:params.pointspread_max-1] .= collect(-params.pointspread_max:-2)
	input_param.r_vector[params.pointspread_max+3:params.pointspread_size] .= collect(2 : params.pointspread_max)
	r_closegame = [1000, 0, 1000]
	input_param.r_vector[params.pointspread_max:params.pointspread_max+2] .= r_closegame
end

# Call update function to initalize SecondParameters according to the values in ExtraParameters
updateSecondParams(params_2)

#Define the Action Space
@enum Action kick two
𝒜=[kick, two];

#2D State Space , [12 drives + termainal state, 81 pointspreads]
𝒮=[[State(d, ps) for d=1:params.num_drives+1 for ps=-params.pointspread_max:params.pointspread_max]...]
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
	return params_2.r_vector[pointspread_final + params.pointspread_max + 1]
end

# Performs the attempt for kick/two, returns the probabilistic outcome
function doAttempt(s::State, a::Action)
	if a == kick
		return 1 * rand(params_2.dist_home_kick, 1)[1]   # Comes in vector form, indexing to extract
	elseif a == two
		return 2 * rand(params_2.dist_home_two, 1)[1]
	end
end

# Rollout a game from the current state. 'result' represents the outcome of the kick/two attempt post-TD ~ [0,1,2].
function SimulateSubgame(s::State, extra_points::Int)
	remaining_drives = params.num_drives+1 - s.drive
	dist_TD_home = params_2.dist_home_two
	dist_TD_opp  = params_2.dist_opp_two
	samples_home = rand(dist_TD_home, remaining_drives)  # Draw drive results for the remaining drives
	samples_opp  = rand(dist_TD_opp , remaining_drives)  # Currently using two-point conversion percentages as p(drive results in TD)
	pointspread_final = s.pointspread + extra_points + 7*sum(samples_home) - 7*sum(samples_opp)

	s′::State = State(params.num_drives+1, pointspread_final)
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
	Pi::Array{Int} = zeros(size(𝒮)[1])
	for i in 1:size(Q)[1]
		if Q[i,1] >= Q[i,2]
			Pi[i] = 1
		else
			Pi[i] = 2
		end
	end
	Pi_reshaped = reshape(Pi, (params.num_drives+1, params.pointspread_size))
	return Pi_reshaped
end

function readCSV(infile::String, home_team::String, away_team::String)
	df = DataFrame(CSV.File(infile))
	team_names = df[:,1]
	team_dict = Dict()
	for (number, name) in enumerate(team_names)
		team_dict[name] = number
	end
	# Extracting data from Data.csv -- [Team name, home kick %, home two %, away two block %]
	params.p_suc_kick  = df[team_dict[home_team], 2]
	params.p_stop_kick = 1 - df[team_dict[home_team], 2]
	params.p_suc_two   = df[team_dict[home_team], 3]
	params.p_stop_two  = df[team_dict[away_team], 4]
end

function computePolicy(n::Int)
	Q = SolveGames(n)
	Pi = GetPolicy(Q)
	return Pi
end

function main(n::Int, home_team::String, away_team::String, output_filepath::String, data_filepath::String="Data/Data.csv", num_drives::Int=12, max_pointspread::Int=10)
	readCSV(data_filepath, home_team, away_team)
	# params.p_suc_kick  = 0.9  # dummy values, actually assigned later using CSV data
	# params.p_stop_kick = 0.1
	# params.p_suc_two   = 0.33
	# params.p_stop_two  = 0.9

	# Secondary Parameter updating
	params.pointspread_max = max_pointspread
	params.num_drives = num_drives
	updateSecondParams(params_2)

	Pi_star = computePolicy(n)
	CSV.write(output_filepath, Tables.table(Pi_star), writeheader=false)
end




##########################################################################################################
# DEBUGGING / TESTING

# CSV Writing
# test_state = State(3,-1)
# test_action = kick
# params.p_suc_kick  = 0.9  # dummy values, actually assigned later using CSV data
# params.p_stop_kick = 0.1
# params.p_suc_two   = 0.005
# params.p_stop_two  = 0.9
# test_Q = SolveGames(1000)
# test_Pi = GetPolicy(test_Q)
# println("Data written")

# CSV Reading - issues with accessing String attributes using String variables
# data_filename = "Data/Data_Transposed.csv"
# home_team = "Arizona_Cardinals"
# away_team = "Atlanta_Falcons"
# readCSV(data_filename, home_team, away_team)

# Full stack testing
home_team = "Arizona Cardinals"
away_team = "Atlanta Falcons"
outfile = "Data/Scenario3_Policies/sample.csv"
main(1000, home_team, away_team, outfile)

# Greenbay Packers vs Tampa Bucks
