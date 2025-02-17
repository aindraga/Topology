using Pkg
Pkg.activate(@__DIR__)

using Distributed, BenchmarkTools, ThreadsX, Parameters
import RobustTDA as rtda



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


max_tasks = Sys.CPU_THREADS
add_workers = 0
current_workers = []
count_procs = nprocs()

if max_tasks > count_procs
    add_workers = max_tasks - (count_procs - 1)
    addprocs(add_workers)
end
println("Number of Workers: $(nworkers())")

@everywhere begin
    using MLDataPattern
    using NearestNeighbors
    using ParallelDataTransfer
    using Statistics

    function make_tree(x)
        return KDTree(x, leafsize=1)
    end
end

function parallel_tree(X)
    m = nworkers()
    Xs = collect(reduce(hcat, fold[2]) for fold in kfolds(shuffleobs(X), m))
    return pmap(make_tree, Xs)
end

function serial_tree(X)
    m = nworkers()
    Xs = collect(reduce(hcat, fold[2]) for fold in kfolds(shuffleobs(X), m))
    return map(make_tree, Xs)
end

# Run once to compile
small_X = myCircle(10, 10);
serial_tree(small_X);
parallel_tree(small_X);

# Data 
X = myCircle(500_000, 500_000);

# Raw Time
serial_res = @timed serial_tree(X);
parallel_res = @timed parallel_tree(X);

# Benchmark Time
b_serial_res = @btimed serial_tree(X);
b_parallel_res = @btimed parallel_tree(X);

begin
    @info "Raw Time:\n $((; serial_res..., value="serial"))"
    @info "Raw Time:\n $((; parallel_res..., value="parallel"))"
    @info "Benchmark Time:\n $((; b_serial_res..., value="serial"))"
    @info "Benchmark Time:\n $((; b_parallel_res..., value="parallel"))"
end
