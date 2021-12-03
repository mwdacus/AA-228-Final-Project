#Plots Win Percentages for Simulations (Scenario 3)
using CSV
using DataFrames
using Plots

cd(@__DIR__)

df1=DataFrame(CSV.File("VI_simulation.csv"))
df2=DataFrame(CSV.File("qlearn_simulation.csv"))
df3=DataFrame(CSV.File("qlearn_simulation1.csv"))

scatter(df1.games,df1.percentage,label="Value Iteration",)
scatter!(df3.games,df3.percentage,label="Q-Learning")


xlabel!("Number of Simulations (games)")
ylabel!("Win probability")
