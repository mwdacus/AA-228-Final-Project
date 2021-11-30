#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Use Value Iteration to plot all probabilities possible combination of probabilities 
#Input: 
    #Rewards
    #Extra point completion percentage (p_suc_kick), 
    #Opponent extra point stopping percentage (p_stop_kick),
    #two point conversion completion percentage (p_suc_two)
    #Opponent two point stopping percentage (p_stop_two)
#Output: two plots of value v. probability of successfully completing that action (different datasets for opponent stopping percentage)
#----------------------------------------------------------------------------------

using POMDPModelTools
using QuickPOMDPs
using POMDPs
using Parameters
using DiscreteValueIteration
using Plots

include("MDPScenario1.jl")

p_home_kick=range(0,1,length=1000)
p_opp_kick=range(0.2,0.8,length=4)
p_home_two=range(0,1,length=1000)
p_opp_two=range(0.2,0.8,length=4)
v_kick=zeros(length(p_opp_kick),length(p_home_kick))
v_two=zeros(length(p_opp_two),length(p_home_two))

for j in 1:length(p_opp_kick)
	for i in 1:length(p_home_kick)
		global params
        local solver
        local policy
		params=ExtraParameters(p_suc_kick=p_home_kick[i],p_stop_kick=p_opp_kick[j],p_suc_two=0,p_stop_two=0)
		solver=ValueIterationSolver(max_iterations=50)
		policy=solve(solver,mdp)
		v_kick[j,i]=value(policy,TD)
	end
end
for j in 1:length(p_opp_two)
	for i in 1:length(p_home_two)
		global params
        local solver
        local policy
		params=ExtraParameters(p_suc_kick=0,p_stop_kick=0,p_suc_two=p_home_two[i],p_stop_two=p_opp_two[j])
		solver=ValueIterationSolver(max_iterations=50)
		policy=solve(solver,mdp)
		v_two[j,i]=value(policy,TD)
	end
end

	plot(p_home_kick,v_kick[1,:],label="p_stop=0.2")
	plot!(p_home_kick,v_kick[2,:],label="p_stop=0.4")
	plot!(p_home_kick,v_kick[2,:],label="p_stop=0.4")
	plot!(p_home_kick,v_kick[3,:],label="p_stop=0.6")
	plot!(p_home_kick,v_kick[4,:],label="p_stop=0.8")
	xlabel!("probability of making extra point")
	ylabel!("value")

	plot(p_home_two,v_two[1,:],label="p_stop=0.2")
	plot!(p_home_two,v_two[2,:],label="p_stop=0.4")
	plot!(p_home_two,v_two[3,:],label="p_stop=0.6")
	plot!(p_home_two,v_two[4,:],label="p_stop=0.8")
	xlabel!("probability of making two point conversion")
	ylabel!("value")

