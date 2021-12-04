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
using DataFramesMeta
using CSV

#Call Scenario 2
include("MDPScenario2.jl")

#Determine the probability the team will score a touchdown (for each team)
p_td=0.3;

#probability of succesfully making extra point
p_kick=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick));
p_two=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two));

#Simulate Game function
function simulate(sarsp,mdp)
    #Assume 12 offensive drives in a game
    for i in 1:12
        #offensive possession
        if rand(Bernoulli(p_td))==true
            global params
            local policy
            if sarsp.s′==[] || sarsp.s′[end]==params.termination_state
                s1=State(6)
            else
                s1=State(6+sarsp.s′[end].PS)
            end
            #call function solver type
            a1=action(VI_policy,s1)
            if a1==kick && rand(Bernoulli(p_kick))==true
                s1′=State(s1.PS+1)
                r1=round(R(s1,a1),digits=3)
            elseif a1==two && rand(Bernoulli(p_two))==true
                s1′=State(s1.PS+2)
                r1=round(R(s1,a1),digits=3)
            else
                s1′=s1
                r1=round(R(s1,a1),digits=3)
            end        
            push!(sarsp,[s1,a1,r1,s1′])
            #update(sarsp,params)
        #opponent possession
        else
            #if opponent scores, assume they go for td, if they get it
            if rand(Bernoulli(p_td))==true
                if sarsp.s′==[] || sarsp.s′[end]==params.termination_state
                    opp=State(-7)
                else
                    opp=State(sarsp.s′[end].PS-7)
                end
            #if they don't
            else
                if sarsp.s′==[] || sarsp.s′[end]==params.termination_state
                    opp=State(0)
                else
                    opp=State(sarsp.s′[end].PS)
                end
            end
            push!(sarsp,[params.termination_state,kick,1,opp])
        end
    end
    # Determine if they win or lose, and add to sars′
    if sarsp.s′[end].PS>0
        s_end=params.win_state
        a_end=kick
        r_end=r1=round(R(s_end,a_end),digits=3)
        s′_end=params.termination_state
    else
        s_end=params.lose_state
        a_end=kick
        r_end=r1=round(R(s_end,a_end),digits=3)
        s′_end=params.termination_state
    end
    push!(sarsp,[s_end,a_end,r_end,s′_end])
    return filter(row ->(row.s′==params.termination_state),sarsp)
  
end

#function for determining average win percentage
function WinPercent(sarsp)
    (row,col)=size(sarsp)
    num_wins=zeros(Int(row/50))
    num_losses=zeros(Int(row/50))
    win_percentage=zeros(Int(row/50))
    counter=1;
    println(size(sarsp))
    for i in 50:50:row
        num_wins[counter]=count(x->(x==params.win_state),sarsp.s[1:i])
        num_losses[counter]=count(x->(x==params.lose_state),sarsp.s[1:i])
        win_percentage[counter]=num_wins[counter]/(num_losses[counter]+num_wins[counter])
        counter+=1
    end
    return win_percentage
end

#main Script Line
#create an empty DataFrame Table (sars′)
sars′=DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);
sars′_VI=DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);
sars′_qlearn=DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);
#state number of game simulations
k=10000;
for j in 1:k
    global sars′_VI
    global sars′_qlearn
    #sars′_VI=simulate(sars′,mdp)
    sars′_qlearn=simulate(sars′,q_mdp)
end
#wins_VI=WinPercent(sars′_VI)
wins_qlearn=WinPercent(sars′_qlearn)
df=DataFrame(games=50:50:k,percentage=wins_qlearn)
CSV.write("Data/Scenario2_Simulations/qlearn_simulation2.csv",df)



