using BenchmarkTools
using RobustTDA
using Distributed
using CSV
using DataFrames

function myCircle(circ_points, noise_points)
	radius = 1.0
	center = (0.0, 0.0)
	angles = 2π * rand(circ_points)
	circ_x_points = center[1] .+ radius * cos.(angles)
	circ_y_points = center[2] .+ radius * sin.(angles)
	x_points = [circ_x_points; 2 .* rand(noise_points) .- 1]
	y_points = [circ_y_points; 2 .* rand(noise_points) .- 1]
	all_points = [[x_points[i], y_points[i]] for i in eachindex(x_points)]
	return all_points
end

function myGrid(s)
	x_grid = y_grid = range(-2, 2, s)
	grid_points = [[x, y] for x in x_grid for y in y_grid]
	return grid_points
end

max_tasks = Sys.CPU_THREADS
add_workers = 0
current_workers = []
count_procs = nprocs()

if max_tasks > count_procs
    add_workers = max_tasks - (count_procs - 1)
    addprocs(add_workers)
end
println("Number of Workers: $(nworkers())")

@everywhere using RobustTDA

function getResults(data_size, grid_size)
    split_points = (data_size / 2) |> Int
	curr_data = myCircle(split_points, split_points)
	curr_grid = myGrid(grid_size)
	nn_res = @benchmark fit($curr_grid, dist($curr_data))
	dtm_res = @benchmark fit($curr_grid, dtm($curr_data, 0.1))
	mom_res = @benchmark fit($curr_grid, momdist($curr_data, nworkers()))
	pmom_res = @benchmark parallel_fit($curr_grid, parallel_momdist($curr_data))
	nn_res = mean(nn_res).time
	dtm_res = mean(dtm_res).time
	mom_res = mean(mom_res).time
	pmom_res = mean(pmom_res).time
	return [nn_res, dtm_res, mom_res, pmom_res, data_size, grid_size]
end

df = DataFrame(NN=Float64[], DTM=Float64[], MoM=Float64[], pMoM=Float64[], data_size=Int64[], grid_size=Int64[])

# 10,000 Data Points
println("Starting 10K")
push!(df, getResults(10_000, 16))
push!(df, getResults(10_000, 32))
push!(df, getResults(10_000, 48))
push!(df, getResults(10_000, 64))
println("Finished 10K")

# 20,000 Data Points
println("Starting 20K")
push!(df, getResults(20_000, 16))
push!(df, getResults(20_000, 32))
push!(df, getResults(20_000, 48))
push!(df, getResults(20_000, 64))
println("Finished 20K")

# 30,000 Data Points
println("Starting 30K")
push!(df, getResults(30_000, 16))
push!(df, getResults(30_000, 32))
push!(df, getResults(30_000, 48))
push!(df, getResults(30_000, 64))
println("Finished 30K")

# 40,000 Data Points
println("Starting 40K")
push!(df, getResults(40_000, 16))
push!(df, getResults(40_000, 32))
push!(df, getResults(40_000, 48))
push!(df, getResults(40_000, 64))
println("Finished 40K")

# 50,000 Data Points
println("Starting 50K")
push!(df, getResults(50_000, 16))
push!(df, getResults(50_000, 32))
push!(df, getResults(50_000, 48))
push!(df, getResults(50_000, 64))
println("Finished 50K")

CSV.write("rtda.csv", df)