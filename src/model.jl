using MLJ
using DataFrames
using Plots
using ZipFile
using UrlDownload
using HTTP

include("src/read.jl")

df = read_data()

#data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
#f = download(data_url)
#z = ZipFile.Reader(f)
#z_by_filename = Dict( f.name => f for f in z.files)
#df = CSV.read(z_by_filename["occurrence.txt"], DataFrame)


