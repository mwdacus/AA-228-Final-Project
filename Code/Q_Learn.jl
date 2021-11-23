#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Based on previous data, use Q-Learning to determine approximate optimal policy for each state
#Input: Incoming Dataset (s,a,r,sp)
#Output: Best Action at each state
#----------------------------------------------------------------------------------

include('main.jl')

#Define Q-Learning Variable
mutable struct QLearning
    𝒮
    𝒜
    γ
    Q
    α
end

model=MDP(S)
#Update Action Value function