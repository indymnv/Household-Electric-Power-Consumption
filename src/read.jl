using DataFrames
using StatsKit
using CSV
using DelimitedFiles

#df = DataFrame(CSV.File("./data/household_power_consumption.txt"), normalizenames=true)
function read_data()
    # Read the text file into a matrix
    data = readdlm("./data/household_power_consumption.txt", ';',)
    # Convert the matrix to a dataframe
    df = DataFrame(data, :auto)
    rename!(df, Symbol.(Vector(df[1,:])))[2:end,:]
    df = df[2:end, :]
    #parsing Floats
    function ensure_floats(arr::Array{Any})
        output_arr = copy(arr)
        for (i, x) in enumerate(arr)
            if !isa(x, Float64)
                output_arr[i] = -1
            end
        end
        return output_arr
    end

    df[:,3:9] = mapcols( ensure_floats, df[!,3:9])

    for i in 3:size(df)[2]
        df[!,i] = convert(Vector{Float64}, df[!, i])
    end
    # Parsing Time
    df[!, :Time] =  parse.(Time, df."Time")
    # parsing Date
    df[!,1] = replace.(df[!,1], "/" => "-")
    df[!,1] = Date.(df[!,1], "d-m-y")
    return df
end


