using Distributed
using BenchmarkTools
@everywhere using MLDataPattern
@everywhere using NearestNeighbors
@everywhere using ParallelDataTransfer
using ThreadsX
using Parameters
import RobustTDA as rtda


max_tasks = Sys.CPU_THREADS
    
add_workers = 0
current_workers = []
count_procs = nprocs()

if max_tasks > count_procs
    add_workers = max_tasks - (count_procs - 1)
    addprocs(add_workers)
end

println("Number of Workers: $(nworkers())")

@with_kw mutable struct DistanceFunction
    k::Integer
    trees::AbstractVector{<:NNTree}
    X::AbstractVector{<:AbstractVector{<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}}
    type::String
    Q::Int64
end

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

@everywhere function processSublist(sublist)
    shuffled_list = shuffleobs(sublist)
    curr_tree = KDTree(reduce(hcat, shuffled_list), leafsize=1)
    return (curr_tree, shuffled_list)
end

function parallelmomdist(dataset)
    dataset_length = length(dataset)
    current_workers = workers()
    count_workers = length(current_workers)

    # Send Folds To Each Worker
    batch_size = dataset_length / count_workers |> floor |> Int
    
    futures = Vector{Any}(undef, count_workers)
    @sync for i in 1:count_workers
        start = (batch_size * (i - 1)) + 1
        if i != count_workers
            curr_lst = dataset[start: batch_size * i]
            curr_future = @spawnat current_workers[i] processSublist(curr_lst)
            futures[i] = curr_future
            continue    
        end
    
        curr_lst = dataset[start:length(dataset)]
        curr_future = @spawnat current_workers[i] processSublist(curr_lst)
        futures[i] = curr_future
    end

    foreach(wait, futures)
    return "finished"

    return DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "momdist",
        Q = count_workers
    ) 
end


# 1,000 Points Results
first_circ = myCircle(750_000, 250_000)

parallel_time = @timed parallelmomdist(first_circ)
mt_time = @timed rtda.momdist(first_circ, 4)

parallel_time = parallel_time.time - parallel_time.compile_time
println("Parallel Approach: ", parallel_time)

mt_time = mt_time.time - mt_time.compile_time
println("Multi-Threading Approach: ", mt_time)
