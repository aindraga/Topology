using Distributed
using BenchmarkTools
@everywhere using MLDataPattern
@everywhere using NearestNeighbors
@everywhere using ParallelDataTransfer
using ThreadsX
using Parameters

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
    # Add workers if needed
    max_tasks = Sys.CPU_THREADS
    
    add_workers = 0
    current_workers = []
    count_procs = nprocs()
    
    if max_tasks > count_procs
        add_workers = max_tasks - count_procs
        current_workers = addprocs(add_workers)
    end

    dataset_length = length(dataset)
    current_workers = workers()
    count_workers = length(current_workers)

    # Send Folds To Each Worker
    batch_size = dataset_length / count_workers |> floor |> Int
    
    for i in 1:count_workers
        start = (batch_size * (i - 1)) + 1
        if i != count_workers
            curr_lst = dataset[start: batch_size * i]
            sendto(current_workers[i], sublist=curr_lst)
            continue    
        end
    
        curr_lst = dataset[start:length(dataset)]
        sendto(current_workers[i], sublist=curr_lst)
    end

    futures = [@spawnat current_workers[i] processSublist(sublist) for i in
                                                                1:count_workers]
    results = fetch.(futures)

    Xq = [item[2] for item in results]
    trees = [item[1] for item in results]

    return nothing

    return DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "momdist",
        Q = count_workers
    ) 

end

function momdist(
    data::AbstractVector{T},
    Q = 0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = ThreadsX.collect(fold[2] for fold in kfolds(shuffleobs(data), Q))

    trees = ThreadsX.collect(KDTree(reduce(hcat, xq), leafsize=1) for xq in Xq)

    return nothing
    DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "momdist",
        Q = Q
    )
end


# 1,000 Points Results
first_circ = myCircle(75000, 25000)

@btime momdist(first_circ, 3)
@btime parallelmomdist(first_circ)

