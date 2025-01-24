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
	using DataFrames;
end

# ╔═╡ 337ae826-ef76-4cc6-86e6-a89af22615c8
md"""# Experiment"""

# ╔═╡ 9e1a71a1-5673-4f88-9a1d-53f5fd6bd956
begin
	num_points_lst = [1000, 3000, 5000]
	grid_sizes_lst = [100, 150, 200]
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
	nn_info_lst = []
	kde_info_lst = []
	dtm_info_lst = []
	for data_size in num_points_lst
		curr_dataset = myCircle(data_size - 100, 100)
		for grid_size in grid_sizes_lst
			curr_grid = myGrid(grid_size)
			
			nn_info = @benchmark myNN($curr_grid, $curr_dataset)
			nn_details = (data_size, grid_size, minimum(nn_info).time, maximum(nn_info).time, median(nn_info).time, mean(nn_info).time)
			
			kde_info = @benchmark myMultiKDE($curr_dataset, $curr_grid)
			kde_details = (data_size, grid_size, minimum(kde_info).time, maximum(kde_info).time, median(kde_info).time, mean(kde_info).time)

			dtm_optimal = Int(floor(sqrt(data_size)))
			dtm_info = @benchmark myDTM($curr_grid, $curr_dataset, $dtm_optimal, 2)
			dtm_details = (data_size, grid_size, minimum(dtm_info).time, maximum(dtm_info).time, median(dtm_info).time, mean(dtm_info).time)
			
			push!(nn_info_lst, nn_details)
			push!(kde_info_lst, kde_details)
			push!(dtm_info_lst, dtm_details)
		end
	end
end

# ╔═╡ 857dda28-00c2-42f6-a072-2f82782a8e61
nn_info_lst

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
# ╠═857dda28-00c2-42f6-a072-2f82782a8e61
