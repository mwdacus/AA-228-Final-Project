#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Based on previous data, use Q-Learning to determine approximate optimal policy for each state
#Input: Incoming Dataset (s,a,r,sp)
#Output: Best Action at each state
#----------------------------------------------------------------------------------

include("MDP.jl")

#Define Q-Learning Variable
mutable struct QLearning
    𝒮
    𝒜
    γ
    Q
    α
end

#Update Action Value function
function update!(model,s,a,r,sp)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a]+=α*(r+γ*maximum(Q[s',:])-Q[s,a])
    return model
end

#Define Additional Variables
α=0.3;
Q=zeros(length(𝒫.𝒮),length(𝒫.𝒜));
k=20;

#Define the Q-Learning Model
model=QLearning(𝒫.𝒮, 𝒫.𝒜, 𝒫.γ, Q, α);

for i in 1:k #Experience Replay
    for j in 1:row #rows in Dataset
        update!(model,D[j,1],D[j,2],D[j,3],D[j,4])
    end
end
