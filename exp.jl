### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ e718db6f-c78d-4df3-a9fd-7ca71adbb45f
begin
	using Pkg
	Pkg.add(url="https://github.com/sidv23/RobustTDA.jl")
	using RobustTDA
	import RobustTDA as rtda
	Pkg.add("Plots")
	using Plots;
	using Statistics;
	Pkg.add("Distances")
	using Distances;
	using LinearAlgebra;
	Pkg.add("BenchmarkTools")
	using BenchmarkTools;
	using DataFrames
end

# ╔═╡ fbf1b521-d562-4ce4-89db-e41201f014e4
begin
	Pkg.add("CSV")
	using CSV
end

# ╔═╡ 337ae826-ef76-4cc6-86e6-a89af22615c8
md"""# Experiment"""

# ╔═╡ 9e1a71a1-5673-4f88-9a1d-53f5fd6bd956
begin
	num_points_lst = [100, 300, 700, 1000]
	grid_sizes_lst = [50, 100, 150, 200]
end

# ╔═╡ faf47d88-39dd-4996-abee-e64113d0e773
function myGrid(s)
	x_grid = y_grid = range(-2, 2, s)
	grid_points = [(x, y) for x in x_grid for y in y_grid]
	return grid_points
end

# ╔═╡ 79805a38-f1ad-4510-ac3c-05201d035787
md"""## My Cubical-Based Functions"""

# ╔═╡ 7cd3d6e6-484a-4955-b5f5-4137a527a8cb
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

# ╔═╡ 72b878d7-02b8-444f-938e-cc559f4e9af5
function myNN(eval_points, data_points)
	dist_mat = pairwise(Euclidean(), eval_points, data_points)
	return minimum(dist_mat, dims=2)
end

# ╔═╡ c100415e-c9f0-4f52-9760-52944a2f3eb9
Kh(r; h=1.0) = exp(-r^2/h)

# ╔═╡ a07ffd32-0405-4ec8-9aeb-2d4ea82b0be8
function myMultiKDE(data_points, eval_points)
	kernel_matrix = pairwise(Euclidean(), eval_points, data_points) .|> (x -> Kh(x; h=0.5))
	results = mean(kernel_matrix, dims=2)
	reshaper = Int32(sqrt(size(eval_points)[1]))
	results = reshape(results, (reshaper, reshaper))
	return results
end

# ╔═╡ 0d504b54-afa4-4e27-bc90-66d9b4c16595
function myDTM(eval_points, data_points, nns, r)
	dist_mat = pairwise(Euclidean(), eval_points, data_points)
	small = sortperm(dist_mat, dims=2)[:, 1:nns]
	vals = dist_mat[small] .^ r
	vals = mean(vals, dims=2) .^ (1 / r)
	reshaper = Int32(sqrt(size(eval_points)[1]))
	vals = reshape(vals, (reshaper, reshaper))
	return vals
end

# ╔═╡ 8acffc55-f015-4cc6-a01c-6b523b01ae74
md"""## Runtime Data Gathering"""

# ╔═╡ 327a70f9-e5af-42cc-a285-db87b2b50960
begin
	my_funcs_runtime_lst = []
	for data_size in num_points_lst
		curr_dataset = myCircle(data_size - 50, 50)
		for grid_size in grid_sizes_lst
			curr_grid = myGrid(grid_size)
			
			nn_info = @benchmark myNN($curr_grid, $curr_dataset)
			nn_details = ("myNN", data_size, grid_size, minimum(nn_info).time, maximum(nn_info).time, median(nn_info).time, mean(nn_info).time)
			
			kde_info = @benchmark myMultiKDE($curr_dataset, $curr_grid)
			kde_details = ("myKDE", data_size, grid_size, minimum(kde_info).time, maximum(kde_info).time, median(kde_info).time, mean(kde_info).time)
			
			dtm_info = @benchmark myDTM($curr_grid, $curr_dataset, 20, 2)
			dtm_details = ("myDTM", data_size, grid_size, minimum(dtm_info).time, maximum(dtm_info).time, median(dtm_info).time, mean(dtm_info).time)
			
			push!(my_funcs_runtime_lst, nn_details)
			push!(my_funcs_runtime_lst, kde_details)
			push!(my_funcs_runtime_lst, dtm_details)
		end
	end
end

# ╔═╡ a978672a-8d7d-402d-a1c8-8c6fc8563c34
my_funcs_runtime_lst

# ╔═╡ db93f704-a7d5-4545-8941-530c51720ddc
begin
	rtda_runtime_data = []
	for dataset_size in num_points_lst
		curr_dataset = myCircle(dataset_size - 100, 100)
		curr_dataset = [[x...] for x in curr_dataset]
		
		rtda_nn_info = @benchmark rtda.dist($curr_dataset)
		rtda_nn_details = ("rtdaNN", dataset_size, 0, minimum(rtda_nn_info).time, maximum(rtda_nn_info).time, median(rtda_nn_info).time, mean(rtda_nn_info).time)
		
		rtda_dtm_info = @benchmark rtda.dtm($curr_dataset, 0.1)
		rtda_dtm_details = ("rtdaDTM", dataset_size, 0, minimum(rtda_dtm_info).time, maximum(rtda_dtm_info).time, median(rtda_dtm_info).time, mean(rtda_dtm_info).time)
		
		rtda_mom_info = @benchmark rtda.momdist($curr_dataset, 100)
		rtda_mom_details = ("rtdaMOM", dataset_size, 0, minimum(rtda_mom_info).time, maximum(rtda_mom_info).time, median(rtda_mom_info).time, mean(rtda_mom_info).time)

		push!(rtda_runtime_data, rtda_nn_details)
		push!(rtda_runtime_data, rtda_dtm_details)
		push!(rtda_runtime_data, rtda_mom_details)
	end
end

# ╔═╡ 80e2fc85-de42-40cd-9922-726813f56c80
my_funcs_df = DataFrame(func=String[], data_size=Int64[], grid_size=Int64[], min_runtime=Float64[], max_runtime=Float64[], med_runtime=Float64[], mean_runtime=Float64[])

# ╔═╡ 00bbd116-6cac-4245-bdd9-790c70021521
my_funcs_df

# ╔═╡ 76bd652a-1190-4ade-aec2-d74e99da1b08
rtda_funcs_df = DataFrame(func=String[], data_size=Int64[], grid_size=Int64[], min_runtime=Float64[], max_runtime=Float64[], med_runtime=Float64[], mean_runtime=Float64[])

# ╔═╡ 44ba70f3-677b-4091-9ff9-14a2e10f6065
rtda_funcs_df

# ╔═╡ d1b8502e-1405-4a16-8c3f-4ba2149d0611
CSV.write("my_funcs_data.csv", my_funcs_df)

# ╔═╡ 884deacc-a08a-4b29-9dea-51093a54659a
CSV.write("rtda_funcs_data.csv", rtda_funcs_df)

# ╔═╡ Cell order:
# ╠═e718db6f-c78d-4df3-a9fd-7ca71adbb45f
# ╠═337ae826-ef76-4cc6-86e6-a89af22615c8
# ╠═9e1a71a1-5673-4f88-9a1d-53f5fd6bd956
# ╠═faf47d88-39dd-4996-abee-e64113d0e773
# ╠═79805a38-f1ad-4510-ac3c-05201d035787
# ╠═7cd3d6e6-484a-4955-b5f5-4137a527a8cb
# ╠═72b878d7-02b8-444f-938e-cc559f4e9af5
# ╠═c100415e-c9f0-4f52-9760-52944a2f3eb9
# ╠═a07ffd32-0405-4ec8-9aeb-2d4ea82b0be8
# ╠═0d504b54-afa4-4e27-bc90-66d9b4c16595
# ╠═8acffc55-f015-4cc6-a01c-6b523b01ae74
# ╠═327a70f9-e5af-42cc-a285-db87b2b50960
# ╠═a978672a-8d7d-402d-a1c8-8c6fc8563c34
# ╠═db93f704-a7d5-4545-8941-530c51720ddc
# ╠═80e2fc85-de42-40cd-9922-726813f56c80
# ╠═00bbd116-6cac-4245-bdd9-790c70021521
# ╠═76bd652a-1190-4ade-aec2-d74e99da1b08
# ╠═44ba70f3-677b-4091-9ff9-14a2e10f6065
# ╠═fbf1b521-d562-4ce4-89db-e41201f014e4
# ╠═d1b8502e-1405-4a16-8c3f-4ba2149d0611
# ╠═884deacc-a08a-4b29-9dea-51093a54659a
