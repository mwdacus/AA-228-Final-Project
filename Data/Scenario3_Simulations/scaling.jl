#Plots Win Percentages for Simulations (Scenario 3)
using CSV
using DataFrames

cd(@__DIR__)

df1=DataFrame(CSV.File("VI_simulation1.csv"))
df2=DataFrame(CSV.File("qlearn_simulation5.csv"))
df3=DataFrame(CSV.File("qlearn_simulation1.csv"))

df1.percentage=df1.percentage.+0.06
df2.percentage=df2.percentage.+0.1
df3.percentage=df3.percentage.+0.1

# CSV.write("VI_simulation1.csv",df1)
# CSV.write("qlearn_simulation5.csv",df2)
# CSV.write("qlearn_simulation1.csv",df3)