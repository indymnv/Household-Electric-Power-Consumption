using Plots 
unicodeplots()
using MLJ
using EvoTrees
using UrlDownload
using ZipFile
using HTTP
using CSV
using Dates
using Statistics
using MLJClusteringInterface
using ParallelKMeans
using Clustering
using FreqTables
using StatsPlots
using RollingFunctions
using StatsBase
using ShiftedArrays
using DataFrames
using Distributions
using HypothesisTests

#read data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
f = download(data_url)
z = ZipFile.Reader(f)
z_by_filename = Dict( f.name => f for f in z.files)
data = CSV.read(z_by_filename["household_power_consumption.txt"], DataFrame,)
describe(data)

#Preprocessing
dropmissing!(data)

for i in 3:8
        data[!,i] = parse.(Float64, data[!,i])
end
data[!,1] = replace.(data[!,1], "/" => "-")
data[!,1] = Date.(data[!,1], "d-m-y")


#Create a variables from dates 
date_time = [DateTime(d, t) for (d,t) in zip(data[!,1], data[!,2])]

data[!,:date_time] = date_time

data[!,:year] = Dates.value.(Year.(data[!,1]))
data[!,:month] = Dates.value.(Month.(data[!,1]))
data[!,:day] = Dates.value.(Day.(data[!,1]))

data[!, :hour] = Dates.value.(Hour.(data[!,2]))
data[!, :minute] = Dates.value.(Minute.(data[!,2]))

data[!, :dayofweek] = [dayofweek(date) for date in data.Date]
data[!, :weekend] = [day in [6, 7] for day in data.dayofweek]


plot([plot(data[1:50000,col]; label = col) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage"]]...)

plot([plot(data[1:50000, :date_time],data[1:50000,col]; label = col, xrot=30) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage", "Sub_metering_1",  "Sub_metering_2", "Sub_metering_3"]]...)

cm = cor(Matrix(data[!,3:9]))
cols = Symbol.(names(data[!,3:9]))

(n,m) = size(cm)
heatmap(cm, 
fc = cgrad([:white,:dodgerblue4]),
xticks = (1:m,cols),
xrot= 90,
size= (500, 500),
yticks = (1:m,cols),
yflip=true)
annotate!([(j, i, text(round(cm[i,j],digits=3),
               8,"Computer Modern",:black))
   for i in 1:n for j in 1:m])

# ╔═╡ 41510658-acc7-4b32-8e79-883093440cf7
md"""
## A brief clustering with kmeans
"""

X = data[!, 3:9]
transformer_instance = Standardizer()
transformer_model = machine(transformer_instance, X)
fit!(transformer_model)
X = MLJ.transform(transformer_model, X);

for m in models()
        println("Model name = ",m.name,", ","Prediction type = ",m.prediction_type,", ","Package name = ",m.package_name);
end

KMeans= @load KMeans pkg=Clustering
kmeans = KMeans(k=3)
#train = collect(Matrix(X)')
mach = machine(kmeans, X) |> fit!

# cluster X into 5 clusters using K-means
#R = machine(train , 5; maxiter=200, display=:iter)
Xsmall = MLJ.transform(mach);
selectrows(Xsmall, 1:4) |> pretty
yhat = MLJ.predict(mach)
data[!,:cluster] = yhat


combine(groupby(data, :cluster), nrow )

#scatterplot with all clusters
scatter(data[1:20000,:].date_time,data[1:20000,:].Voltage,  group=data[1:20000,:].cluster,)

#Scatter plot for several figures
plot([scatter(data[1:20000, :date_time],data[1:20000,col]; group=data[1:20000,:].cluster, size=(1200, 1000), title = col, xrot=30) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage", "Sub_metering_1",  "Sub_metering_2", "Sub_metering_3"]]...)

#Density plot
p1 = @df data density(:Global_active_power, group = (:cluster), legend = :topright, title = "Active Power")
p2 = @df data density(:Global_reactive_power, group = (:cluster), legend = :topright, title = "Rective Power")
p3 = @df data density(:Global_intensity, group = (:cluster), legend = :topright , title = "Intensity")
p4 = @df data density(:Voltage, group = (:cluster), legend = :topright, title = "Voltage")
p5 = @df data density(:Sub_metering_1, group = (:cluster), legend = :topright, title = "Sub_1")
p6 = @df data density(:Sub_metering_2, group = (:cluster), legend = :topright, title = "Sub_2")

plot(p1, p2, p3, p4, p5, p6, layout=(3,2), legend=true)

#boxplot
b1 =@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2, title ="Global active power")
b2 =@df data boxplot(string.(:cluster), :Global_reactive_power, fillalpha=0.75, linewidth=2, title = "Global reactive power")
b3 = @df data boxplot(string.(:cluster), :Global_intensity, fillalpha=0.75, linewidth=2, title ="Global intensity")
b4 = @df data boxplot(string.(:cluster), :Voltage, fillalpha=0.75, linewidth=2, title = "Voltage")
plot(b1, b2, b3, b4 ,layout=(2,2), legend=false)

#heatmap
h1 =heatmap(freqtable(data,:cluster,:dayofweek)./freqtable(data,:cluster), title = "day of week")
h2 =heatmap(freqtable(data,:cluster,:hour)./freqtable(data,:cluster), title = "hour")
h3 = heatmap(freqtable(data,:cluster,:month)./freqtable(data,:cluster), title = "month")
h4 = heatmap(freqtable(data,:cluster,:day)./freqtable(data,:cluster), title = "day")

plot(h1, h2, h3, h4 ,layout=(2,2), legend=false)

#feature engineering
data[!, :lag_30] = Array(ShiftedArray(data.Voltage, 30))
replace!(data.lag_30, missing => 0);

#split the data
train = copy(filter(x -> x.Date < Date(2010,10,01), data))
test = copy(filter(x -> x.Date >= Date(2010,10,01), data))

#Drop coluns
select!(train, Not([:Date, :Time, :date_time, :cluster, ]))
select!(test, Not([:Date, :Time, :date_time, :cluster, ]))

#get the y data
y_train = copy(train[!,:Voltage])
y_test = copy(test[!,:Voltage])

#cyclical encoder for dates based var
function cyclical_encoder(df::DataFrame, columns::Union{Array, Symbol}, max_val::Union{Array, Int} )
for (column, max) in zip(columns, max_val)
            #max_val = maximum(df[:, column])
    df[:, Symbol(string(column) * "_sin")] = sin.(2*pi*df[:, column]/max)
    df[:, Symbol(string(column) * "_cos")] = cos.(2*pi*df[:, column]/max)
    end
    return df
end

columns_selected = [:day, :year, :month, :hour, :minute, :dayofweek]
max_val = [31, 2010, 12, 23, 59, 7]
train_cyclical = cyclical_encoder(train, columns_selected, max_val)
test_cyclical = cyclical_encoder(test, columns_selected, max_val)

names(train_cyclical)

#Coerced the variables
train_coerced = coerce(train_cyclical, 
        :year_sin=>Continuous,
        :month_sin=>Continuous,
        :day_sin=>Continuous,
        :hour_sin=>Continuous,
        :minute_sin=>Continuous,
        :dayofweek_sin=>Continuous,
        :year_cos=>Continuous,
        :month_cos=>Continuous,
        :day_cos=>Continuous,
        :hour_cos=>Continuous,
        :minute_cos=>Continuous,
        :dayofweek_cos=>Continuous,
        :lag_30=>Continuous,
        :weeekend=>Multiclass,
        :interval_day=>Multiclass,
);

test_coerced = coerce(test_cyclical, 
        :year_sin=>Continuous,
        :month_sin=>Continuous,
        :day_sin=>Continuous,
        :hour_sin=>Continuous,
        :minute_sin=>Continuous,
        :dayofweek_sin=>Continuous,
        :year_cos=>Continuous,
        :month_cos=>Continuous,
        :day_cos=>Continuous,
        :hour_cos=>Continuous,
        :minute_cos=>Continuous,
        :dayofweek_cos=>Continuous,
        :lag_30=>Continuous,
        :weeekend=>Multiclass,
        :interval_day=>Multiclass,
);

train_coerced

EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees verbosity=1
etr = EvoTreeRegressor(max_depth =15)

machreg = machine(etr, train_coerced[!,14:end], y_train);

fit!(machreg);
#predict and get error
pred_etr = MLJ.predict(machreg, test_coerced[!,14:end]);
rms_score = rms(pred_etr, y_test)

#Error
error = y_test - pred_etr

# Estimate distribution parameters:
mean_err = mean(pred_etr)
std_err = std(pred_etr)
# Create QQ plot:
qqplot(Normal(mean_err, std_err), pred_etr, title = "QQ-plot error distribution")

#get plot for error
er1 = histogram( y_test - pred_etr, title = "error rms $rms_score", bins= 30)

er2 = scatter( y_test , pred_etr, )
plot(er1, er2, layout=(1,2), legend=false)

#also contribute with line plots
plot(y_test, label = "real")
plot!(pred_etr, label= "predict")
