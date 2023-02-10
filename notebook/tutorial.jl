### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ a4ba2c57-8cf2-4bb7-800a-4839af64849c
begin
	using DataFrames
	using Plots
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
end

# ╔═╡ 34f74a05-818c-4cdf-ac7c-a74b7a478329
md"""
# Using evotrees.jl for time series prediction
"""

# ╔═╡ 9be44075-dc61-4957-a761-e54315f91d2b
md"""
Let's import our libraries
"""

# ╔═╡ 09d06610-9b0e-428e-8b99-55f8bf0de376
md"""
###### In this notebook I want to explore and show you how you can use evotrees for a time series prediction. At the same time, this notebooks provide different methods to get a feature engineering elements to improve the model's performance
"""

# ╔═╡ 1c41d476-dda7-45b8-bc25-ec757244f932
begin
	data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
	f = download(data_url)
	z = ZipFile.Reader(f)
	z_by_filename = Dict( f.name => f for f in z.files)
	data = CSV.read(z_by_filename["household_power_consumption.txt"], DataFrame,)
end

# ╔═╡ 3b7dd1cd-f2f7-4229-b0c4-4cdf7aea495a
#begin
#	include("../src/read.jl")
#	data = read_data()
#end

# ╔═╡ 652f39e8-227c-4f6c-a9f0-7b576e8f89e8
describe(data)

# ╔═╡ 62e02377-da4c-4381-9e59-4f32372f4fb5
dropmissing!(data)

# ╔═╡ a4526d2f-5895-4468-aa43-1192b2dd50b5
# This activate later when come back to reading from source

begin
	for i in 3:8
		data[!,i] = parse.(Float64, data[!,i])
	end
	data[!,1] = replace.(data[!,1], "/" => "-")
    data[!,1] = Date.(data[!,1], "d-m-y")
end

# ╔═╡ b20e5a78-3c5c-4ae8-b49c-38d3eaf1bba1
data

# ╔═╡ 2364d69b-f093-4fa0-a366-96c4091660bf
begin
	#Create a variable 
	date_time = [DateTime(d, t) for (d,t) in zip(data[!,1], data[!,2])]
	
	data[!,:date_time] = date_time
	
	#Create variable for date
	data[!,:year] = Dates.value.(Year.(data[!,1]))
	data[!,:month] = Dates.value.(Month.(data[!,1]))
	data[!,:day] = Dates.value.(Day.(data[!,1]))
	
	#Create variable for time
	data[!, :hour] = Dates.value.(Hour.(data[!,2]))
	data[!, :minute] = Dates.value.(Minute.(data[!,2]))
	
	#Create variable for weekends
	data[!, :dayofweek] = [dayofweek(date) for date in data.Date]
	data[!, :weekend] = [day in [6, 7] for day in data.dayofweek]
end

# ╔═╡ 82f8cfc4-41e2-479a-8f2e-d60c8e4bf23d
data

# ╔═╡ 0100421c-dea1-4ec4-a622-b96ac3a9775e
plot([plot(data[1:50000,col]; label = col) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage"]]...)

# ╔═╡ 185e8de6-b136-45d3-9603-7cb62fe46a95
plot([plot(data[1:50000, :date_time],data[1:50000,col]; label = col, xrot=30) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage", "Sub_metering_1",  "Sub_metering_2", "Sub_metering_3"]]...)

# ╔═╡ d90786bf-aab1-49bf-8b7b-82827c61da1b
begin
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
end

# ╔═╡ 41510658-acc7-4b32-8e79-883093440cf7
md"""
# A brief clustering with kmeans
"""

# ╔═╡ fc4bd69b-0fb4-46b4-b1a5-fb95bb4990d5
begin
	X = data[!, 3:9]
	transformer_instance = Standardizer()
	transformer_model = machine(transformer_instance, X)
	fit!(transformer_model)
	X = MLJ.transform(transformer_model, X);
end

# ╔═╡ e7f713c8-d56b-4b00-9a65-ba1b6411999f
for m in models()
    println("Model name = ",m.name,", ","Prediction type = ",m.prediction_type,", ","Package name = ",m.package_name);
end

# ╔═╡ f5cff6d6-3652-496c-afdf-b6bb4a70d203
begin
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
	#@assert nclusters(R) == 5 # verify the number of clusters
	
	#a = assignments(R) # get the assignments of points to clusters
	#c = counts(R) # get the cluster sizes
	#M = R.centers # get the cluster centers
end

# ╔═╡ 9cadf7ab-428b-4c92-8aed-2995bc13b629
combine(groupby(data, :cluster), nrow )

# ╔═╡ e3239243-ccd3-403a-be05-24ee8c43b766
scatter(data[1:20000,:].date_time,data[1:20000,:].Voltage,  group=data[1:20000,:].cluster,)

# ╔═╡ e34c100f-5107-4933-abed-ae25ad16d662
plot([scatter(data[1:20000, :date_time],data[1:20000,col]; group=data[1:20000,:].cluster, size=(1200, 1000), title = col, xrot=30) for col in ["Global_active_power",  "Global_reactive_power", "Global_intensity", "Voltage", "Sub_metering_1",  "Sub_metering_2", "Sub_metering_3"]]...)

# ╔═╡ 6a3e5035-445a-4f5d-9b0d-cedd251f2b6a
begin
	p1 = @df data density(:Global_active_power, group = (:cluster), legend = :topright, title = "Active Power")
	p2 = @df data density(:Global_reactive_power, group = (:cluster), legend = :topright, title = "Rective Power")
	p3 = @df data density(:Global_intensity, group = (:cluster), legend = :topright , title = "Intensity")
	p4 = @df data density(:Voltage, group = (:cluster), legend = :topright, title = "Voltage")
	p5 = @df data density(:Sub_metering_1, group = (:cluster), legend = :topright, title = "Sub_1")
	p6 = @df data density(:Sub_metering_2, group = (:cluster), legend = :topright, title = "Sub_2")
	
	plot(p1, p2, p3, p4, p5, p6, layout=(3,2), legend=true)
end

# ╔═╡ 4e5e989f-9cfa-4b04-87a4-9490a66d0c0d
begin
	b1 =@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2, title ="Global active power")
	b2 =@df data boxplot(string.(:cluster), :Global_reactive_power, fillalpha=0.75, linewidth=2, title = "Global reactive power")
	b3 = @df data boxplot(string.(:cluster), :Global_intensity, fillalpha=0.75, linewidth=2, title ="Global intensity")
	b4 = @df data boxplot(string.(:cluster), :Voltage, fillalpha=0.75, linewidth=2, title = "Voltage")

	#@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2)
	#@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2)
	plot(b1, b2, b3, b4 ,layout=(2,2), legend=false)
end

# ╔═╡ 3b1f7548-4e85-40b2-b6d9-1dfcba6e49ab
begin
	h1 =heatmap(freqtable(data,:cluster,:dayofweek)./freqtable(data,:cluster), title = "day of week")
	h2 =heatmap(freqtable(data,:cluster,:hour)./freqtable(data,:cluster), title = "hour")
	h3 = heatmap(freqtable(data,:cluster,:month)./freqtable(data,:cluster), title = "month")
	h4 = heatmap(freqtable(data,:cluster,:day)./freqtable(data,:cluster), title = "day")

	#@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2)
	#@df data boxplot(string.(:cluster), :Global_active_power, fillalpha=0.75, linewidth=2)
	plot(h1, h2, h3, h4 ,layout=(2,2), legend=false)
end

# ╔═╡ d5c80f35-b750-41c1-8a30-9cb89c59f5aa
data[!, :lag_30] = Array(ShiftedArray(data.Global_active_power, 30))

# ╔═╡ 2f295092-b0fe-441b-996b-3414be28fcc0
replace!(data.lag_30, missing => 0);

# ╔═╡ 2ac2439e-b679-405d-9103-1a9b38c36200
length(data.lag_30)

# ╔═╡ 9375e4a3-3ba0-4e4b-a47f-23edb9676ded
rollstd(data.lag_30, 30)

# ╔═╡ b734f136-c680-4ee0-8896-79d2241b3099
data

# ╔═╡ d29bbac3-ee6d-47af-b8db-ab99f5b030b4
begin
	train = copy(filter(x -> x.Date < Date(2010,10,01), data))
	test = copy(filter(x -> x.Date >= Date(2010,10,01), data))
end

# ╔═╡ 60895046-09c1-4cc7-8528-35470e7eba09
begin
	select!(train, Not([:Date, :Time, :date_time, :cluster, ]))
	select!(test, Not([:Date, :Time, :date_time, :cluster, ]))
end

# ╔═╡ 0483f035-c3fa-4de4-be98-a73f430f89d3
train

# ╔═╡ 3b811c60-a918-43fd-bd14-1e0bad0aba1f
begin
	y_train = copy(train[!,:Voltage])
	y_test = copy(test[!,:Voltage])
end

# ╔═╡ 2de3e03c-c4e6-46d8-8b89-e929e35cd4b3
function cyclical_encoder(df::DataFrame, columns::Union{Array, Symbol}, max_val::Union{Array, Int} )
    for (column, max) in zip(columns, max_val)
		#max_val = maximum(df[:, column])
        df[:, Symbol(string(column) * "_sin")] = sin.(2*pi*df[:, column]/max)
        df[:, Symbol(string(column) * "_cos")] = cos.(2*pi*df[:, column]/max)
    end
    return df
end

# ╔═╡ 6610f46e-5475-4865-b690-cdde061b467e
begin
	columns_selected = [:day, :year, :month, :hour, :minute, :dayofweek]
	max_val = [31, 2010, 12, 23, 59, 7]
	train_cyclical = cyclical_encoder(train, columns_selected, max_val)
	test_cyclical = cyclical_encoder(test, columns_selected, max_val)
end

# ╔═╡ 43172899-ccd9-48d6-b3fd-ed9a3add8833
names(train_cyclical)

# ╔═╡ 6e1a56b7-69cd-46e3-9716-22f0bda1f2f7
begin
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
end

# ╔═╡ 13935b3c-b7b1-496c-b62b-cec377b4512f
train_coerced

# ╔═╡ 95667564-9d8a-45ca-a5c8-b5baad187f4b
begin
	EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees verbosity=0
	etr = EvoTreeRegressor(max_depth =15)
	
	machreg = machine(etr, train_coerced[!,14:end], y_train);

	fit!(machreg);
end

# ╔═╡ 8b65a47d-8717-4ce0-85dc-353c9dcb16b2
begin
	# 0.7725232666581258
	pred_etr = MLJ.predict(machreg, test_coerced[!,14:end]);
	rms_score = rms(pred_etr, y_test)
end

# ╔═╡ abca48a9-c9d0-4726-a3df-b0801371241a
begin
	er1=histogram( y_test - pred_etr, title = "error rms $rms_score", bins= 20)
	er2 = scatter( y_test , pred_etr, )
	
	
	plot(er1, er2, layout=(1,2), legend=false)
	
end

# ╔═╡ 8fc811e6-8644-4d66-bc50-a49b4da56d64
begin
	plot(y_test, label = "real")
	plot!(pred_etr, label= "predict")
end

# ╔═╡ Cell order:
# ╟─34f74a05-818c-4cdf-ac7c-a74b7a478329
# ╟─9be44075-dc61-4957-a761-e54315f91d2b
# ╟─09d06610-9b0e-428e-8b99-55f8bf0de376
# ╠═a4ba2c57-8cf2-4bb7-800a-4839af64849c
# ╠═1c41d476-dda7-45b8-bc25-ec757244f932
# ╠═3b7dd1cd-f2f7-4229-b0c4-4cdf7aea495a
# ╠═652f39e8-227c-4f6c-a9f0-7b576e8f89e8
# ╠═62e02377-da4c-4381-9e59-4f32372f4fb5
# ╠═a4526d2f-5895-4468-aa43-1192b2dd50b5
# ╠═b20e5a78-3c5c-4ae8-b49c-38d3eaf1bba1
# ╠═2364d69b-f093-4fa0-a366-96c4091660bf
# ╠═82f8cfc4-41e2-479a-8f2e-d60c8e4bf23d
# ╠═0100421c-dea1-4ec4-a622-b96ac3a9775e
# ╠═185e8de6-b136-45d3-9603-7cb62fe46a95
# ╠═d90786bf-aab1-49bf-8b7b-82827c61da1b
# ╠═41510658-acc7-4b32-8e79-883093440cf7
# ╠═fc4bd69b-0fb4-46b4-b1a5-fb95bb4990d5
# ╠═e7f713c8-d56b-4b00-9a65-ba1b6411999f
# ╠═f5cff6d6-3652-496c-afdf-b6bb4a70d203
# ╠═9cadf7ab-428b-4c92-8aed-2995bc13b629
# ╠═e3239243-ccd3-403a-be05-24ee8c43b766
# ╠═e34c100f-5107-4933-abed-ae25ad16d662
# ╠═6a3e5035-445a-4f5d-9b0d-cedd251f2b6a
# ╠═4e5e989f-9cfa-4b04-87a4-9490a66d0c0d
# ╠═3b1f7548-4e85-40b2-b6d9-1dfcba6e49ab
# ╠═d5c80f35-b750-41c1-8a30-9cb89c59f5aa
# ╠═2f295092-b0fe-441b-996b-3414be28fcc0
# ╠═2ac2439e-b679-405d-9103-1a9b38c36200
# ╠═9375e4a3-3ba0-4e4b-a47f-23edb9676ded
# ╠═b734f136-c680-4ee0-8896-79d2241b3099
# ╠═d9ff8fd1-f621-4a8b-a7ac-037ceee4d9ad
# ╠═739a1dc5-7a3e-4cef-8841-e91d729dbc1b
# ╠═d29bbac3-ee6d-47af-b8db-ab99f5b030b4
# ╠═60895046-09c1-4cc7-8528-35470e7eba09
# ╠═0483f035-c3fa-4de4-be98-a73f430f89d3
# ╠═3b811c60-a918-43fd-bd14-1e0bad0aba1f
# ╠═2de3e03c-c4e6-46d8-8b89-e929e35cd4b3
# ╠═6610f46e-5475-4865-b690-cdde061b467e
# ╠═43172899-ccd9-48d6-b3fd-ed9a3add8833
# ╠═6e1a56b7-69cd-46e3-9716-22f0bda1f2f7
# ╠═13935b3c-b7b1-496c-b62b-cec377b4512f
# ╠═95667564-9d8a-45ca-a5c8-b5baad187f4b
# ╠═8b65a47d-8717-4ce0-85dc-353c9dcb16b2
# ╠═8a186f25-06e7-458c-9d43-7a1d495c7a4d
# ╠═abca48a9-c9d0-4726-a3df-b0801371241a
# ╠═8fc811e6-8644-4d66-bc50-a49b4da56d64
