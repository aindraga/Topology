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
	Pkg.add("DataFrames")
	using DataFrames;
	using Statistics;
	Pkg.add("Distances")
	using Distances;
	using LinearAlgebra;
end

# ╔═╡ 9caebf0b-5219-4f34-adca-850975144ad2
begin
	# Signal from a circle with noise
	n, m = 500, 100
	N = n + m
	signal = 2 .* rtda.randCircle(n, sigma=0.05)
	noise = rtda.randMClust(m, a=1, b=1, λ_parent=2, λ_child=100, r=0.1)
	X = [signal; noise]
	Xn = [[x...] for x in Tuple.(eachrow(X))]
	
	# Distance-like Functions
	f_dist, f_dtm, f_momdist = rtda.dist(Xn), rtda.dtm(Xn, 0.1), rtda.momdist(Xn, 2m + 1)
end

# ╔═╡ 57fd19ac-6385-4a6c-a975-c6e8c0fcd6d5
Xn

# ╔═╡ 2f8bdb39-74b9-4da9-99d7-11f6a15519a6
begin
	xseq = -3:0.1:3
	yseq = -3:0.1:3
	plt = f -> plot(xseq, yseq, (x,y) -> RobustTDA.fit([[x,y]],f)[1], st=:surface)
end

# ╔═╡ 06a3e110-a6d7-4df6-9035-0988885b83cf
begin
	plotly()
	f = f_momdist
	plt(f)
end

# ╔═╡ 337ae826-ef76-4cc6-86e6-a89af22615c8
md"""# Experiment"""

# ╔═╡ 9e1a71a1-5673-4f88-9a1d-53f5fd6bd956
begin
	num_points_lst = collect(range(500, 2000, step=500))
	grid_sizes = collect(range(100, 500, step=100))
end

# ╔═╡ 758c09bb-9e11-45bd-a65d-f4ca4ebbb8d7
grid_sizes

# ╔═╡ faf47d88-39dd-4996-abee-e64113d0e773
function createGrid(s)
	x_grid = [x for x in range(0, 5, s) for i in range(0, 5, s)]
	y_grid = [y for i in range(0, 5, s) for y in range(0, 5, s)]
	grid_points = [x_grid y_grid]
	return grid_points
end

# ╔═╡ 79805a38-f1ad-4510-ac3c-05201d035787
md"""## My Cubical-Based Functions"""

# ╔═╡ 7cd3d6e6-484a-4955-b5f5-4137a527a8cb
function createCircle(num_points, num_noise)
	radius = 1.0
	center = (0.0, 0.0)
	angles = 2π * rand(num_points)
	circ_x_points = center[1] .+ radius * cos.(angles)
	circ_y_points = center[2] .+ radius * sin.(angles)
	x_points = [circ_x_points; 2 .* rand(num_noise) .- 1]
	y_points = [circ_y_points; 2 .* rand(num_noise) .- 1]
	return [x_points y_points]
end

# ╔═╡ 72b878d7-02b8-444f-938e-cc559f4e9af5
function shortestDistances(eval_points, base_points)
	dist_mat = pairwise(Euclidean(), eval_points, base_points)
	return minimum(dist_mat, dims=2)
end

# ╔═╡ a07ffd32-0405-4ec8-9aeb-2d4ea82b0be8
function multikde(data::Matrix{Float64}, points, kernel_func::Function, H::Union{Matrix{Float64}, Nothing} = nothing)

	n, d = size(data)
	if H === nothing
		sigmas = std(data, dims=1)
		factor = (4 / (d + 2))^(1 / (d + 4)) * n^(-1 / (d + 4))
		H = zeros(d, d)
		diags = sigmas * factor
		H[diagind(H)] = diags
	end
	
	if size(H) != (d, d)
		return false
	end
	
	if all(eigvals(H) .>= 0) == false
		return false
	end

	det_H = det(H)
	inv_H = inv(H)
	
	densities = zeros(size(points)[1])
	normalization = 1 / (sqrt(det_H) * (2π)^(d / 2))
	data = Array(data')
	for (i, x) in enumerate(eachrow(points))
		data_comp = x .- data
		u_mat = inv_H * data_comp
		kern_aspect = kernel_func.(eachcol(u_mat))
		sum_kernel = sum(kern_aspect)
		densities[i] = normalization * sum_kernel / n
	end

	return densities
end

# ╔═╡ df613a6f-84ce-4a0a-8fae-255e00166e6d
function gaussian_kernel(u)
    return exp(-0.5 * dot(u, u))
end

# ╔═╡ 0d504b54-afa4-4e27-bc90-66d9b4c16595
function dtm(eval_points, base_points, nn, r)
	dist_mat = pairwise(Euclidean(), eval_points', base_points')
	small = sortperm(dist_mat, dims=1)[1:nn, :]
	vals = dist_mat[small] .^ r
	return mean(vals, dims=1) .^ (1 / r)
end

# ╔═╡ da883445-2915-4642-88df-bb1788933a14
all_grids = [createGrid(gr) for gr in grid_sizes]

# ╔═╡ 9965ea73-d791-431a-98cc-0baea59dd5fd
all_circs = [createCircle(cr, 100) for cr in num_points_lst]

# ╔═╡ 8acffc55-f015-4cc6-a01c-6b523b01ae74
md"""## Runtime Data Gathering"""

# ╔═╡ beff736b-37fa-4bb7-a787-88624e6ffebb
short_dist_times = [("my_shortest_dist", size(c_circ)[1] - 100, sqrt(size(c_grid)[1]), mean([@elapsed shortestDistances(c_grid', c_circ') for i in 1:3:1])) for c_circ in all_circs for c_grid in all_grids]

# ╔═╡ c17e1218-230e-4128-b278-1ab54147d662
kde_times = [("my_kde", size(c_circ)[1] - 100, sqrt(size(c_grid)[1]), mean([@elapsed multikde(c_circ, c_grid, gaussian_kernel) for i in 1:3:1])) for c_circ in all_circs for c_grid in all_grids]

# ╔═╡ 323a3636-e276-40a5-9482-d48870f39868
dtm_times = [("my_dtm", size(c_circ)[1] - 100, sqrt(size(c_grid)[1]), mean([@elapsed dtm(c_grid, c_circ, 25, 2) for i in 1:3:1])) for c_circ in all_circs for c_grid in all_grids]

# ╔═╡ 56640b54-ef03-4e7e-9245-8687c31f25d7
new_circs = [[[c...] for c in Tuple.(eachrow(c_circ))] for c_circ in all_circs]

# ╔═╡ b7588163-518d-4d5b-b53d-508bca2b7fe1
s_dist_times = [("rtda_shortest_dist", size(c_circ)[1] - 100, 0, mean([@elapsed rtda.dist(c_circ) for i in 1:10:1])) for c_circ in new_circs]

# ╔═╡ d43f040c-9062-4611-bbbd-f6cf71537f71
s_dtm_times = [("rtda_dtm", size(c_circ)[1] - 100, 0, mean([@elapsed rtda.dtm(c_circ, 0.1) for i in 1:10:1])) for c_circ in new_circs]

# ╔═╡ e7cfbd0c-e307-458f-96ba-b2d0c240ff61
s_mom_times = [("rtda_mom", size(c_circ)[1] - 100, 0, mean([@elapsed rtda.momdist(c_circ, 201) for i in 1:10:1])) for c_circ in new_circs]

# ╔═╡ 97127db0-f268-4e6f-82cd-e9070660a825
all_rows = [short_dist_times; kde_times; dtm_times; s_dist_times; s_dtm_times; s_mom_times]

# ╔═╡ ff962bed-b894-4a27-9947-c826a4df8520
begin
	df = DataFrame(dist_func=String[], data_points=Int[], grid_size=Int[], time=Int[])
	df = vcat(df, DataFrame(all_rows, [:dist_func, :data_points, :grid_size, :time]))
end

# ╔═╡ Cell order:
# ╠═e718db6f-c78d-4df3-a9fd-7ca71adbb45f
# ╠═9caebf0b-5219-4f34-adca-850975144ad2
# ╠═57fd19ac-6385-4a6c-a975-c6e8c0fcd6d5
# ╠═2f8bdb39-74b9-4da9-99d7-11f6a15519a6
# ╠═06a3e110-a6d7-4df6-9035-0988885b83cf
# ╠═337ae826-ef76-4cc6-86e6-a89af22615c8
# ╠═9e1a71a1-5673-4f88-9a1d-53f5fd6bd956
# ╠═758c09bb-9e11-45bd-a65d-f4ca4ebbb8d7
# ╠═faf47d88-39dd-4996-abee-e64113d0e773
# ╠═79805a38-f1ad-4510-ac3c-05201d035787
# ╠═7cd3d6e6-484a-4955-b5f5-4137a527a8cb
# ╠═72b878d7-02b8-444f-938e-cc559f4e9af5
# ╠═a07ffd32-0405-4ec8-9aeb-2d4ea82b0be8
# ╠═df613a6f-84ce-4a0a-8fae-255e00166e6d
# ╠═0d504b54-afa4-4e27-bc90-66d9b4c16595
# ╠═da883445-2915-4642-88df-bb1788933a14
# ╠═9965ea73-d791-431a-98cc-0baea59dd5fd
# ╠═8acffc55-f015-4cc6-a01c-6b523b01ae74
# ╠═beff736b-37fa-4bb7-a787-88624e6ffebb
# ╠═c17e1218-230e-4128-b278-1ab54147d662
# ╠═323a3636-e276-40a5-9482-d48870f39868
# ╠═56640b54-ef03-4e7e-9245-8687c31f25d7
# ╠═b7588163-518d-4d5b-b53d-508bca2b7fe1
# ╠═d43f040c-9062-4611-bbbd-f6cf71537f71
# ╠═e7cfbd0c-e307-458f-96ba-b2d0c240ff61
# ╠═97127db0-f268-4e6f-82cd-e9070660a825
# ╠═ff962bed-b894-4a27-9947-c826a4df8520
