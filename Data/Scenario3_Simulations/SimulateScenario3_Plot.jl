#Plots Win Percentages for Simulations (Scenario 3)
using CSV
using DataFrames
using Plots

cd(@__DIR__)

# df1=DataFrame(CSV.File("VI_simulation.csv"))
df2=DataFrame(CSV.File("VI_simulation1.csv"))
df3=DataFrame(CSV.File("qlearn_simulation1.csv"))

scatter(df2.games,df2.percentage,label="Value Iteration",)
scatter!(df3.games,df3.percentage,label="Q-Learning")


xlabel!("Number of Simulations (games)")
ylabel!("Win probability")
ylims!((0.59,0.69))
yticks!([0.6,0.62,0.64,0.6, 0.68])

savefig("10000_plot.png")