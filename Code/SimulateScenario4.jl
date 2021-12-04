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
# cd(@__DIR__)

#Call Scenario 2
include("MDPScenario4.jl")

#Determine the probability the team will score a touchdown (for each team)
p_td=0.3;

#probability of succesfully making extra point
p_kick=mean.(params.p_suc_kick)*(1-mean.(params.p_stop_kick));
p_two=mean.(params.p_suc_two)*(1-mean.(params.p_stop_two));

#Simulate Game function
function simulate(sarsp,mdp)
    #Assume 12 offensive drives in a game
    # println("RESTART GAME")
    # println()
    for i in 1:params.num_drives
        # println("Drive: ", i)
        # TD scored on drive
        if rand(Bernoulli(p_td))==true
            # println("Touchdown!")
            global params
            local policy
            # Home team
            if !(iseven(i))
                # println("home possession")
                if i==1 || sarsp.s′==[]
                    s1 = State(i, 6)
                else
                    s1 = State(i, sarsp.s′[end].pointspread + 6)
                end
            
                #call function solver type
                # a1 = kick
                # a1 = two
                a1 = action(VI_policy,s1)
                # a1 = action(q_learning_policy, s1)

                if a1==kick && rand(Bernoulli(p_kick))==true
                    s1′ = State(i+1, s1.pointspread + 1)
                elseif a1==two && rand(Bernoulli(p_two))==true
                    s1′ = State(i+1, s1.pointspread + 2)
                else
                    s1′ = State(i+1 , s1.pointspread) 
                end

                r1=round(R(s1,a1),digits=3)   
                push!(sarsp,[s1,a1,r1,s1′])
            # Opponent possession
            else 
                # println("away possession")
                if i==1 || sarsp.s′==[] || sarsp.s′[end]==params.termination_state
                    s1 = State(i, -6)
                else
                    s1 = State(i, sarsp.s′[end].pointspread - 6)
                end
                # Assume they go for a kick
                if rand(Bernoulli(p_kick))==true
                    # println("good kick")
                    s1′ = State(i+1, s1.pointspread - 1)
                else
                    # println("missed kick")
                    s1′ = State(i+1, s1.pointspread)
                end

                a1 = kick
                r1 = 1
                push!(sarsp,[s1,a1,r1,s1′])
            end
        # TD not scored on possession
        else
            a1 = kick
            r1 = 1
            if sarsp.s′ == []
                s1 = State(i, 0)
                s1′ = State(i+1, 0)
            else
                s1 = State(i, sarsp.s′[end].pointspread)
                s1′ = State(i+1, sarsp.s′[end].pointspread)
            end
            push!(sarsp, [s1, a1, r1, s1′])
        end
    end

    # Determine if they win or lose, and add to sars′
    if sarsp.s′[end].pointspread > 0  # HOKAJ - adding '=' will change the percentage
        s_end = params.win_state
        a_end = kick
        r_end = r1 =round(R(s_end,a_end),digits=3)
        s′_end = params.termination_state
        # push!(sarsp,[s_end,a_end,r_end,s′_end])

    elseif sarsp.s′[end].pointspread < 0
        s_end = params.lose_state
        a_end = kick
        r_end = r1 = round(R(s_end,a_end),digits=3)
        s′_end = params.termination_state
        # push!(sarsp,[s_end,a_end,r_end,s′_end])
    else  # Say a tie is 50/50 win/loss, for math purposes
        if rand(Bernoulli(0.99))
            s_end = params.win_state
        else
            s_end = params.lose_state
        end
        a_end = kick
        r_end = r1 = round(R(s_end,a_end),digits=3)
        s′_end = params.termination_state
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
sars′        = DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);
sars′_VI     = DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);
sars′_qlearn = DataFrame(s=State[],a=Action[],r=Float64[],s′=State[]);

# VALUE ITERATION
#state number of game simulations
k = 10000;
for j in 1:k
    global sars′_VI
    sars′_VI=simulate(sars′,mdp)

    if isinteger(j/1000)
        println(j)
    end
end
wins_VI=WinPercent(sars′_VI)
df=DataFrame(games=50:50:k,percentage=wins_VI)
# CSV.write("Data/Scenario3_Simulations/VI_simulation01.csv",df)
# CSV.write("Data/Scenario3_Simulations/VI_simulation1.csv",df)
# CSV.write("Data/Scenario3_Simulations/VI_simulation10.csv",df)
CSV.write("Data/Scenario3_Simulations/VI_simulation100.csv",df)


# Q-LEARNING
#state number of game simulations
# k = 10000;
# for j in 1:k
#     global sars′_qlearn
#     sars′_qlearn=simulate(sars′,q_mdp)
#     if isinteger(j/1000)
#         println(j)
#     end
# end
# wins_qlearn=WinPercent(sars′_qlearn)
# df=DataFrame(games=50:50:k,percentage=wins_qlearn)
# # CSV.write("Data/Scenario3_Simulations/qlearn_simulation5.csv",df)  # 50 case
# # CSV.write("Data/Scenario3_Simulations/qlearn_simulation1.csv",df)  # 100 case
# println("csv written!!!")