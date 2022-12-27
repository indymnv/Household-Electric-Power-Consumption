using Plots: Legend
using DataFrames
using StatsKit
using CSV
using DelimitedFiles
using Plots
using StatsPlots
using Statistics

include("src/read.jl")

#read data
df = real_data()

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

#Create variable for weekends
df[!, :dayofweek] = [dayofweek(date) for date in df.Date]
df[!, :weekend] = [day in [6, 7] for day in df.dayofweek]

plot(df[!,3], size = (1500,1000))

#Plot for simple sample of time series
p = plot(layout=(2,3), size = (1000,800))
for j in 3:9
    for i in unique(df[!,:year])
        df_year = filter(row ->row.year == i,df)
        display(plot!(p[j-2], 
            df_year[1:1000,j] ,
            label = "$i",
            title = names(df)[j],
        ))
    end
end    

#plot for average by months
num_cols = names(df, findall(x -> eltype(x) <: Number, eachcol(df)))
p = plot(layout=(2,3), size = (1000, 800))
for j in 3:9
    for i in unique(df[!,:year])[2:end]
        df_year = filter(row ->row.year == i,df)
        mean_df = sort(combine(groupby(df_year, ["month"]), 
                num_cols .=> mean .=> num_cols), :month)
        display(plot!(p[j-2], 
            mean_df[!,j] ,
            label = "$i",
            title = names(df_year)[j],
        ))
    end
end   

# Plot difference between every month
fig = plot(layout = (1,7), size = (1500, 600))
for col in 3:9
    name_col = names(df)[col]
    df2 = unstack(sort(df[!, ["date_time","month", name_col]], :month), "month", name_col)
    df2 = df2[!,2:end]
    for i in 1:12
        display(@df df2 StatsPlots.boxplot!(fig[col-2], collect(skipmissing(df2[!,i])), label= false, title = name_col,))
    end
end

# Compare weekday and weekends
fig = plot(layout = (1,7), size = (1500, 600))
for col in 3:9
    name_col = names(df)[col]
    df2 = sort(df[!, ["date_time", "weekend", name_col]],)
    df2 = df2[!,2:end]
    display(@df filter(x -> x.weekend == true, df2) StatsPlots.boxplot!(fig[col-2], 
            filter(x -> x.weekend == true, df2)[!, name_col], label= false, title = name_col,))
    display(@df filter(x -> x.weekend == false, df2) StatsPlots.boxplot!(fig[col-2],
            filter(x -> x.weekend == false, df2)[!, name_col], label= false, title = name_col,))
end


# Plot for season average
# Plot by hour average consumption boxplot can be nice
fig = plot(layout = (1,7), size = (1500, 600))
for col in 3:9
    name_col = names(df)[col]
    df2 = unstack(sort(df[!, ["date_time","hour", name_col]], :hour), "hour", name_col)
    df2 = df2[!,2:end]
    for i in 1:24
        display(@df df2 StatsPlots.boxplot!(fig[col-2], collect(skipmissing(df2[!,i])), label= false, title = name_col,))
    end
end


# monthly average energy consumptions (by days)
fig = plot(layout = (1,7), size = (1500, 600))
for col in 3:9
    name_col = names(df)[col]
    df2 = unstack(sort(df[!, ["date_time","dayofweek", name_col]], :dayofweek), "dayofweek", name_col)
    df2 = df2[!,2:end]
    for i in 1:7
        display(@df df2 StatsPlots.boxplot!(fig[col-2], collect(skipmissing(df2[!,i])), label= false, title = name_col,))
    end
end


# Create a average rolling and plot the trends again (every days or wathever)
# Use Dickey-Fuller test (I don't now why)

plot([StatsPlots.histogram(df[!,col]; label = col) for col in ["Sub_metering_1", 
                    "Sub_metering_2", "Sub_metering_3", "Voltage"]]...)
