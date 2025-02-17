using Distributed
using MLDataPattern

max_tasks = Sys.CPU_THREADS
current_worker_count = nworkers()

if current_worker_count < max_tasks
    additional_workers = max_tasks - (current_worker_count - 1)
    addprocs(additional_workers)
end

println("Number of Workers: $(nworkers())")

@everywhere function pairwise_sums(arr)
    n = length(arr)
    result = Array{Int}(undef, n, n)  # Create a new n x n array
    for i in 1:n
        for j in 1:n
            result[i, j] = arr[i] + arr[j]
        end
    end
    return result
end

function split_sort(lst)
    all_folds = [item[2] for item in kfolds(lst, nworkers())]
    all_workers = workers()
    results = Vector{Matrix{Int64}}()
    @sync for i in eachindex(all_folds)
        curr_fold = all_folds[i]
        curr_future = @spawnat all_workers[i] pairwise_sums(curr_fold)
        @async begin
           curr_result = fetch(curr_future)
           push!(results, curr_result) 
        end
    end
    return results
end


my_arr = rand(1:1_000, 10_000)

println("Length Dataset: $(length(my_arr))")

parallel_time = @timed split_sort(my_arr)
println("Parallel Time: $(parallel_time.time - parallel_time.compile_time)")

sync_time = @timed pairwise_sums(my_arr)
println("Synchronous Time: $(sync_time.time - sync_time.compile_time)")