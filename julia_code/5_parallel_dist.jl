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
    import RobustTDA as rtda
    import RobustTDA: DistanceFunction

    function make_tree(x)
        return KDTree(x, leafsize=1)
    end

    function tree_dist(tree, X)
        return knn(tree, X, 1)[2] |> Base.Flatten |> collect
    end
end

function parallel_momdist(X)
    m = nworkers()
    Xq = collect(reduce(hcat, fold[2]) for fold in kfolds(shuffleobs(X), m))
    trees = pmap(make_tree, Xq)
    return DistanceFunction(
        k=1,
        trees=trees,
        X=[X],
        type="momdist",
        Q=m
    )
end

function serial_momdist(X)
    m = nworkers()
    Xq = collect(reduce(hcat, fold[2]) for fold in kfolds(shuffleobs(X), m))
    trees = map(make_tree, Xq)
    return rtda.DistanceFunction(
        k=1,
        trees=trees,
        X=[X],
        type="momdist",
        Q=m
    )
end

function serial_fit(X, df)
    X_mat = reduce(hcat, X)
    dists = hcat(map(tree -> tree_dist(tree, X_mat), df.trees)...)
    return median(dists, dims=2)
end


function parallel_fit(X, df)
    X_mat = reduce(hcat, X)
    dists = hcat(pmap(tree -> tree_dist(tree, X_mat), df.trees)...)
    return median(dists, dims=2)
end

# Run once to compile
small_X = myCircle(100, 100);
serial_fit(small_X, serial_momdist(small_X));
parallel_fit(small_X, parallel_momdist(small_X));

# Data 
X = myCircle(400_000, 400_000);
# Raw Time
serial_res = @timed serial_fit(X, serial_momdist(X));
parallel_res = @timed parallel_fit(X, parallel_momdist(X));

# Benchmark Time
b_serial_res = @btimed serial_fit(X, serial_momdist(X));
b_parallel_res = @btimed parallel_fit(X, parallel_momdist(X));

begin
    @info "Raw Time:\n $((; serial_res..., value="serial"))\n\n"
    @info "Raw Time:\n $((; parallel_res..., value="parallel"))\n\n"
    @info "Benchmark Time:\n $((; b_serial_res..., value="serial"))\n\n"
    @info "Benchmark Time:\n $((; b_parallel_res..., value="parallel"))\n\n"
end

# Ns = [10, 100, 1000, 10_000, 100_000]
# serial_times = zeros(length(Ns))
# parallel_times = zeros(length(Ns))
# using Plots
# using ProgressMeter
# for (i, N) in enumerate(Ns)
#     X = myCircle(N, N)
#     t1 = @btimed serial_fit(X, serial_momdist(X))
#     t2 = @btimed parallel_fit(X, parallel_momdist(X))
#     serial_times[i] = t1.time
#     parallel_times[i] = t2.time
# end
# plot(Ns, serial_times, label="Serial", xlabel="N", ylabel="Time (s)", title="Serial vs Parallel Median Distance")
# plot!(Ns, parallel_times, label="Parallel")