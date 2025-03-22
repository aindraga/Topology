# Imports
using BenchmarkTools
using Distances
using DataFrames
using CSV

# Dataset Generation
function myCircle(circ_points, noise_points)
	radius = 1.0
	center = (0.0, 0.0)
	angles = 2π * rand(circ_points)
	circ_x_points = center[1] .+ radius * cos.(angles)
	circ_y_points = center[2] .+ radius * sin.(angles)
	x_points = [circ_x_points; 2 .* rand(noise_points) .- 1]
	y_points = [circ_y_points; 2 .* rand(noise_points) .- 1]
	all_points = [(x_points[i], y_points[i]) for i in eachindex(x_points)]
	return all_points
end

# Grid Creation
function myGrid(s)
	x_grid = y_grid = range(-2, 2, s)
	grid_points = [(x, y) for x in x_grid for y in y_grid]
	return grid_points
end

# My Distance Functions
function myNN(eval_points, data_points)
	dist_mat = pairwise(Euclidean(), eval_points, data_points)
	return minimum(dist_mat, dims=2)
end

Kh(r; h) = exp(-r/h)

function myMultiKDE(data_points, eval_points)
	kernel_matrix = pairwise(Euclidean(), eval_points, data_points) .|> (x -> Kh(x; h=0.5))
	results = mean(kernel_matrix, dims=2)
	return results
end

function myDTM(eval_points, data_points, nns, r)
	dist_mat = pairwise(Euclidean(), eval_points, data_points)
	small = sortperm(dist_mat, dims=2)[:, 1:nns]
	vals = dist_mat[small] .^ r
	vals = mean(vals, dims=2) .^ (1 / r)
	return vals
end

function getResults(data_size, grid_size)
	split_size = (data_size / 2) |> Int
	curr_data = myCircle(split_size, split_size)
	curr_grid = myGrid(grid_size)
	nn_res = @benchmark myNN($curr_grid, $curr_data)
	kde_res = @benchmark myMultiKDE($curr_data, $curr_grid)
	dtm_res = @benchmark myDTM($curr_grid, $curr_data, 10, 2)
	nn_time = mean(nn_res).time
	kde_time = mean(kde_res).time
	dtm_time = mean(dtm_res).time
	return [nn_time, kde_time, dtm_time, data_size, grid_size]
end

# 1,000 Data Points
println("Starting 1K")
res_1k_16g = getResults(1_000, 16)
res_1k_32g = getResults(1_000, 32)
res_1k_64g = getResults(1_000, 64)
println("Finished 1K")

# 10,000 Data Points
println("Starting 10K")
res_10k_16g = getResults(10_000, 16)
res_10k_32g = getResults(10_000, 32)
res_10k_64g = getResults(10_000, 64)
println("Finished 10K")

# 25,000 Data Points 
println("Starting 25K")
res_25k_16g = getResults(25_000, 16)
res_25k_32g = getResults(25_000, 32)
res_25k_64g = getResults(25_000, 64)
println("Finished 25K")

# 50,000 Data Points
println("Starting 50K")
res_50k_16g = getResults(50_000, 16)
res_50k_32g = getResults(50_000, 32)
res_50k_64g = getResults(50_000, 64)
println("Finished 50K")

# 100,000 Data Points
println("Starting 100K")
res_100k_16g = getResults(100_000, 16)
res_100k_32g = getResults(100_000, 32)
res_100k_64g = getResults(100_000, 64)
println("Finished 100K")

df = DataFrame(NN=Float64[], KDE=Float64[], DTM=Float64[], data_size=Int64[], grid_size=Int64[])

push!(df, res_1k_16g)
push!(df, res_1k_32g)
push!(df, res_1k_64g)

push!(df, res_10k_16g)
push!(df, res_10k_32g)
push!(df, res_10k_64g)

push!(df, res_25k_16g)
push!(df, res_25k_32g)
push!(df, res_25k_64g)

push!(df, res_50k_16g)
push!(df, res_50k_32g)
push!(df, res_50k_64g)

push!(df, res_100k_16g)
push!(df, res_100k_32g)
push!(df, res_100k_64g)

CSV.write("ctda.csv", df)