#Plots Win Percentages for Simulations (Scenario 3)
using CSV
using DataFrames
using Plots

cd(@__DIR__)

df1=DataFrame(CSV.File("VI_simulation01.csv"))
df2=DataFrame(CSV.File("qlearn_simulation5.csv"))
df3=DataFrame(CSV.File("qlearn_simulation1.csv"))

scatter(df1.games,df1.percentage,label="Value Iteration (100)",)
scatter!(df2.games,df2.percentage,label="Q-Learning (50)")
scatter!(df3.games,df3.percentage,label="Q-Learning (100)")


# df1=DataFrame(CSV.File("VI_simulation01.csv"))
# df2=DataFrame(CSV.File("VI_simulation1.csv"))
# df3=DataFrame(CSV.File("VI_simulation10.csv"))
# df4=DataFrame(CSV.File("VI_simulation100.csv"))
# scatter(df1.games,df1.percentage.-0.12,label="Value Iteration (1)",)
# scatter!(df2.games,df2.percentage.-0.05,label="Value Iteration (10)")
# scatter!(df3.games,df3.percentage.-0.00,label="Value Iteration (100)")
# scatter!(df4.games,df4.percentage.+0.015,label="Value Iteration (1000)")


xlabel!("Number of Simulations (games)")
ylabel!("Win probability")
# ylims!((0.63,0.73))
# yticks!([0.6,0.62,0.64,0.6, 0.68])

savefig("scenario3_VI_comparison.png")