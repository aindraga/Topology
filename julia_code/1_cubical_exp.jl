# Imports
using Pkg
using BenchmarkTools
import RobustTDA as rtda
using Distances
Pkg.instantiate()

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

# Data Size 1,000, Grid Size 32x32
curr_data = myCircle(500, 500)
curr_grid = myGrid(32)

nn_1k_32 = @benchmark myNN(curr_grid, curr_data)
kde_1k_32 = @benchmark myMultiKDE(curr_data, curr_grid)
dtm_1k_32 = @benchmark myDTM(curr_grid, curr_data, 20, 2)

println(nn_1k_32)
println(kde_1k_32)
println(dtm_1k_32)

curr_grid = myGrid(64)

nn_1k_64 = @benchmark myNN(curr_grid, curr_data)
kde_1k_64 = @benchmark myMultiKDE(curr_data, curr_grid)
dtm_1k_64 = @benchmark myDTM(curr_grid, curr_data, 20, 2)

println(nn_1k_64)
println(kde_1k_64)
println(dtm_1k_64)

curr_grid = myGrid(128)

nn_1k_128 = @benchmark myNN(curr_grid, curr_data)
kde_1k_128 = @benchmark myMultiKDE(curr_data, curr_grid)
dtm_1k_128 = @benchmark myDTM(curr_grid, curr_data, 20, 2)

println(nn_1k_128)
println(kde_1k_128)
println(dtm_1k_128)
