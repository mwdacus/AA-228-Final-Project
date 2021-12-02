#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Simulate Game time Scenario
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
using DataFrames
using CSV

include("MDPScenario2.jl")
#Determine the probability the team will score a touchdown (for each team)
p_td=0.5;
p_kick=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick));
p_two=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two));

#Call Scenario 2

#create an empty DataFrame Table (sars′)
sars′=DataFrame(s=State[],a=Action[],r=Int[],s′=State[])

#Simulate Game
function simulate(sarsp,mdp)
    #Assume 12 offensive drives in a game
    for i in 1:12
        #offensive possession
        if rand(Bernoulli(p_td))==true
            global params
            local solver
            local policy
            if sarsp.s==[]
                s1=State(6)
            else
                s1=State(6+sarsp.s′[end].PS)
            end
            solver=ValueIterationSolver(max_iterations=100)
            policy=solve(solver,mdp)
            a1=action(policy,s1)
            if a1==kick && rand(Bernoulli(p_kick))==true
                s1′=State(s1.PS+1)
                r1=R(s1,a1,s1′)
            elseif a1==two && rand(Bernoulli(p_two))==true
                s1′=State(s1.PS+2)
                r1=R(s1,a1,s1′)
            else
                s1′=s1
                r1=0
            end        
            push!(sarsp,[s1,a1,r1,s1′])
            #update(sarsp,params)
        #opponent possession
        else
            #if opponent scores, assume they go for td, if they get it
            if rand(Bernoulli(p_td))==true
                if sarsp.s′==[]
                    opp=State(-7)
                else
                    opp=State(sarsp.s′[end].PS-7)
                end
            #if they don't
            else
                if sarsp.s′==[]
                    opp=State(0)
                else
                    opp=State(sarsp.s′[end].PS)
                end
            end
            push!(sarsp,[State(0),kick,1,opp])
        end
    end
    return sarsp
end

#=
function update(sarsp,params)
	if sarsp.s′[end]==kickmade
		params.p_suc_kick=Beta(params.p_suc_kick.α+1,params.p_suc_kick.β)
		params.p_stop_kick=Beta(params.p_stop_kick.α,params.p_stop_kick.β)
	elseif sarsp.s′[end]==twomade
		params.p_suc_two=Beta(params_suc_two.α+1,params.p_suc_two.β)
		params.p_stop_two=Beta(params_stop_two.α,params.p_stop_two.β)
	elseif sarsp.s′[end]==missed && a==kick
		params.p_suc_kick=Beta(params.p_suc_kick.α,params.p_suc_kick.β+1)
		params.p_stop_kick=Beta(params.p_stop_kick.α+1,params.p_stop_kick.β)
	else
		params.p_suc_two=Beta(params.p_suc_two.α,params.p_suc_two.β+1)
		params.p_stop_two=Beta(params.p_stop_two.α+1,params.p_stop_two.β)
	end
end
=#


simulate(sars′,mdp)
sars′


CSV.write("test.csv",sars′)