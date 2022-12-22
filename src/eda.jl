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

p = plot(layout=(2,3))
for j in 3:9
    for i in unique(df[!,:year])
        df_year = filter(row ->row.year == i,df)
        display(plot!(p[j-2], 
            df_year[1:1000,j] ,
            label = "$i",
            title = names(df)[j],
            size = (1000, 800)
        ))
    end
end    #plot!(df[2:550,4])

p = plot(layout=(2,3))
for j in 3:9
    for i in unique(df[!,:year])
        df_year = filter(row ->row.year == i,df)
        display(plot!(p[j-2], 
            df_year[1:1000,j] ,
            label = "$i",
            title = names(df)[j],
            size = (1000, 800)
        ))
    end
end    #plot!(df[2:550,4])


plot( )

byyear =  groupby(df, :year)


plot(df[!,4] )
