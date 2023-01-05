using MLJ
using DataFrames
using Plots
using ZipFile
using Statistics
using ParallelKMeans

include("src/read.jl")


df = read_data()

#data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
#f = download(data_url)
#z = ZipFile.Reader(f)
#z_by_filename = Dict( f.name => f for f in z.files)
#df = CSV.read(z_by_filename["household_power_consumption.txt"], DataFrame)

#X = df[!,3:end]
X = df[!, 3:9]
transformer_instance = Standardizer()
transformer_model = machine(transformer_instance, X)
fit!(transformer_model)
X = MLJ.transform(transformer_model, X);


kmeans = @load KMedoids pkg=Clustering

train = collect(Matrix(X)')

# cluster X into 5 clusters using K-means
R = kmeans(train , 5; maxiter=200, display=:iter)

@assert nclusters(R) == 5 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers







train = collect(Matrix(df[:, 3:end])')

# cluster X into 20 clusters using K-means
R = kmeans(train , 5; maxiter=200, display=:iter)

@assert nclusters(R) == 5 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers


