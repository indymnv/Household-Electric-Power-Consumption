using DataFrames
using StatsKit
using CSV
using DelimitedFiles
using Plots

include("src/read.jl")

#read data
df = read_data()

#Create a variable 
date_time = [DateTime(d, t) for (d,t) in zip(df[!,1], df[!,2])]

df[!,:date_time] = date_time

#Create variable for date
df[!,:year] = Year.(df[!,1])
df[!,:month] = Month.(df[!,1])
df[!,:day] = Day.(df[!,1])

#Create variable for time
df[!, :hour] = Hour.(df[!,2])
df[!, :minute] = Minute.(df[!,2])

plot(df[2:1000, 10], df[2:1000,3], )
#plot!(df[2:550,4])

