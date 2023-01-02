using MLJ
using DataFrames
using Plots
using ZipFile

include("src/read.jl")


df = read_data()

#data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
#f = download(data_url)
#z = ZipFile.Reader(f)
#z_by_filename = Dict( f.name => f for f in z.files)
#df = CSV.read(z_by_filename["household_power_consumption.txt"], DataFrame)

X = df[!,3:end]

train = collect(Matrix(df[:, 3:end])')

# cluster X into 20 clusters using K-means
R = kmeans(train , 5; maxiter=200, display=:iter)

@assert nclusters(R) == 5 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers


