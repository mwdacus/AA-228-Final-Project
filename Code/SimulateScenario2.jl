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

#Determine the probability the team will score a touchdown (for each team)
p_td=0.5;
p_kick=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick));
p_two=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two));

#Call Scenario 2
include("MDPScenario2.jl")

#create an empty DataFrame Table (sars′)
sars′=DataFrame(s=State[],a=Action[],r=Int[],s′=State[])

#Simulate Game
function simulate(sarsp,mdp)
    s=State(0)
    #Assume 12 offensive drives in a game
    for i in 1:12
        #offensive possession
        if rand(Bernoulli(p_td))==true
            s.PS=6+s.PS
            global params
            local solver
            local policy
            solver=ValueIterationSolver(max_iterations=100)
            policy=solve(solver,mdp)
            a=action(policy,s)
            if a==kick && rand(Bernoulli(p_kick))==true
                s′=State(s.PS+1)
                r=R(s,a,s′)
            elseif a==two && rand(Bernoulli(p_two))==true
                s′=State(s.PS+2)
                r=R(s,a,s′)
            else
                s′=s
                r=0
            end        
            push!(sarsp,[s,a,r,sp])
            update(sarsp,params)
        #opponent possession
        else
            #if opponent scores, assume they go for td
            if rand(Bernoulli(p_td))==true
                s.PS=-7*i
            else
                pass
            end
        end
    end
    return sarsp
end


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



simulate(sars′,mdp)