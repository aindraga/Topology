### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7b8ab4fe-d903-11ef-03e4-a3f76c7e5a6e
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

# ╔═╡ 0b8959a5-b4f0-405c-b43d-0a017977e98a
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

# ╔═╡ 5829e8bf-ccfa-4f3a-8573-7dcdf5b6feb0
function getNNStats(dataset)
	nn_info = @benchmark rtda.dist($dataset)
	return nn_info
end

# ╔═╡ d1ff167f-e4bf-4c7b-9ee8-a5ad59f4ad82
function getDTMStats(dataset)
	optimal_prop = sqrt(length(dataset)) / length(dataset)
	dtm_info = @benchmark rtda.dtm($dataset, $optimal_prop)
	return dtm_info
end

# ╔═╡ 9051062d-ea75-4e01-8021-3ab18b5fffd5
function getMOMStats(dataset)
	mom_info = @benchmark rtda.momdist($dataset)
	return mom_info
end

# ╔═╡ 4381972f-01c1-468e-9fac-cd22e4d7eeb9
md"""# Size 1000 Results"""

# ╔═╡ 482d2e30-c862-4634-8383-b7a72eb46943
begin
	first_circ = myCircle(750, 250)
	first_circ = [[x...] for x in first_circ]
end

# ╔═╡ 1511370c-79ba-4118-87ca-10bf87e0d632
first_nn_stats = getNNStats(first_circ)

# ╔═╡ 188e4e53-45ab-4026-91b7-eb1459a89338
first_dtm_stats = getDTMStats(first_circ)

# ╔═╡ 507f6647-8256-4020-962c-ac9f2f334dde
first_mom_stats = getMOMStats(first_circ)

# ╔═╡ 29eb8353-acd9-4c8f-acff-39f53e74f984
md"""# Size 2000 Results"""

# ╔═╡ 4dbe223c-34d5-4880-842b-a19b8889fbc4
begin
	second_circ = myCircle(1500, 500)
	second_circ = [[x...] for x in second_circ]
end

# ╔═╡ 3b75d92e-1449-457a-b1a8-fd1146f33eba
second_nn_stats = getNNStats(second_circ)

# ╔═╡ 02d48702-00e8-4afd-ba3a-ab89d0197374
second_dtm_stats = getDTMStats(second_circ)

# ╔═╡ ab63eb21-b36a-4d00-93c6-3a5e7c276575
second_mom_stats = getMOMStats(second_circ)

# ╔═╡ 8eb7bb16-9a26-4990-860d-3644a35a5a75
md"""# Size 3000 Results"""

# ╔═╡ 98f66b36-dddd-467b-a889-5c3f5fdbc179
begin
	third_circle = myCircle(2250, 750)
	third_circle = [[x...] for x in third_circle]
end

# ╔═╡ d5a899bb-845a-497c-9fcf-7d152908c691
third_nn_stats = getNNStats(third_circle)

# ╔═╡ 385b5192-dcb7-4965-a0e1-a4315e8255ad
third_dtm_stats = getDTMStats(third_circle)

# ╔═╡ 97c040d5-11f5-4fce-a9b1-b8c54aa897b8
third_mom_stats = getMOMStats(third_circle)

# ╔═╡ 45941da5-bbd4-4cfa-9220-5c5d92d3a70f
md"""# Size 4000 Results"""

# ╔═╡ f088b720-590b-42d5-b3d0-abb147c90720
begin
	fourth_circle = myCircle(3000, 1000)
	fourth_circle = [[x...] for x in fourth_circle]
end

# ╔═╡ 88f960ce-fece-4ed9-8f61-c8e979c12e2c
fourth_nn_stats = getNNStats(fourth_circle)

# ╔═╡ d9825f3c-834b-4af7-9023-19f93f1c29f4
fourth_dtm_stats = getDTMStats(fourth_circle)

# ╔═╡ 691de595-9c9f-4be8-bfd7-4459ed4a9362
fourth_mom_stats = getMOMStats(fourth_circle)

# ╔═╡ d0a066a6-6b79-4aa0-8dad-287cb1026d55
md"""# Size 5000 Results"""

# ╔═╡ 7e93c05b-57ea-4ddc-b87b-141e0a2ac437
begin
	fifth_circle = myCircle(3750, 1250)
	fifth_circle = [[x...] for x in fifth_circle]
end

# ╔═╡ f552bb4b-789a-403e-9e2d-6d5a1becb913
fifth_nn_stats = getNNStats(fifth_circle)

# ╔═╡ 9b8302d6-8f48-4695-a901-2cafacaadbfc
fifth_dtm_stats = getDTMStats(fifth_circle)

# ╔═╡ 24a36630-a3c4-482b-be21-ad704abb87e9
fifth_mom_stats = getMOMStats(fifth_circle)

# ╔═╡ e359a3b5-745d-4424-8ed5-80e948cca4e4
begin
	nn_stats = [first_nn_stats, second_nn_stats, third_nn_stats, fourth_nn_stats, fifth_nn_stats]
	nn_means = [mean(item).time for item in nn_stats]
	nn_stds = [std(item).time for item in nn_stats]
end

# ╔═╡ 20c1dd23-7445-4c79-93b1-e985ed9f355c
begin
	dtm_stats = [first_dtm_stats, second_dtm_stats, third_dtm_stats, fourth_dtm_stats, fifth_dtm_stats]
	dtm_means = [mean(item).time for item in dtm_stats]
	dtm_stds = [std(item).time for item in dtm_stats]
end

# ╔═╡ c6ffd653-9907-4695-8f1c-0edbb03b2d19
begin
	mom_stats = [first_mom_stats, second_mom_stats, third_mom_stats, fourth_mom_stats, fifth_mom_stats]
	mom_means = [mean(item).time for item in mom_stats]
	mom_stds = [mean(item).time for item in mom_stats]
end

# ╔═╡ cd17ec88-246b-43fe-8975-9790c2bc2d10
num_points = [i * 1000 for i in 1:5]

# ╔═╡ dad5c54d-e361-4e47-a82e-96394fefcdce
begin
	plot(num_points, log.(nn_means), label="Shortest Distance",
	title="Log Runtimes for RTDA Functions", xlabel="Number of Data Points",
	ylabel="Log-Scaled Runtime (Log ns)", )
	plot!(num_points, log.(dtm_means), label="DTM")
	plot!(num_points, log.(mom_means), label="MoM")
end

# ╔═╡ 597c4b7b-f912-4232-9bff-f021ab3064c1
begin
	nn_mem_stats = [mean(item).memory for item in nn_stats]
	dtm_mem_stats = [mean(item).memory for item in dtm_stats]
	mom_mem_stats = [mean(item).memory for item in mom_stats]
end

# ╔═╡ bc1c40e9-dd17-4516-bb0b-22252fafce6b
begin
	plot(num_points, log.(nn_mem_stats), title="Log Memory Estimates for RTDA Functions", xlabel="Number of Data Points", ylabel = "Log-Scaled Memory Estimates (Log Bytes)", label="Shortest Distance")
	plot!(num_points, log.(dtm_mem_stats), label="DTM")
	plot!(num_points, log.(mom_mem_stats), label="MoM")
end

# ╔═╡ Cell order:
# ╠═7b8ab4fe-d903-11ef-03e4-a3f76c7e5a6e
# ╠═0b8959a5-b4f0-405c-b43d-0a017977e98a
# ╠═5829e8bf-ccfa-4f3a-8573-7dcdf5b6feb0
# ╠═d1ff167f-e4bf-4c7b-9ee8-a5ad59f4ad82
# ╠═9051062d-ea75-4e01-8021-3ab18b5fffd5
# ╠═4381972f-01c1-468e-9fac-cd22e4d7eeb9
# ╠═482d2e30-c862-4634-8383-b7a72eb46943
# ╠═1511370c-79ba-4118-87ca-10bf87e0d632
# ╠═188e4e53-45ab-4026-91b7-eb1459a89338
# ╠═507f6647-8256-4020-962c-ac9f2f334dde
# ╠═29eb8353-acd9-4c8f-acff-39f53e74f984
# ╠═4dbe223c-34d5-4880-842b-a19b8889fbc4
# ╠═3b75d92e-1449-457a-b1a8-fd1146f33eba
# ╠═02d48702-00e8-4afd-ba3a-ab89d0197374
# ╠═ab63eb21-b36a-4d00-93c6-3a5e7c276575
# ╠═8eb7bb16-9a26-4990-860d-3644a35a5a75
# ╠═98f66b36-dddd-467b-a889-5c3f5fdbc179
# ╠═d5a899bb-845a-497c-9fcf-7d152908c691
# ╠═385b5192-dcb7-4965-a0e1-a4315e8255ad
# ╠═97c040d5-11f5-4fce-a9b1-b8c54aa897b8
# ╠═45941da5-bbd4-4cfa-9220-5c5d92d3a70f
# ╠═f088b720-590b-42d5-b3d0-abb147c90720
# ╠═88f960ce-fece-4ed9-8f61-c8e979c12e2c
# ╠═d9825f3c-834b-4af7-9023-19f93f1c29f4
# ╠═691de595-9c9f-4be8-bfd7-4459ed4a9362
# ╠═d0a066a6-6b79-4aa0-8dad-287cb1026d55
# ╠═7e93c05b-57ea-4ddc-b87b-141e0a2ac437
# ╠═f552bb4b-789a-403e-9e2d-6d5a1becb913
# ╠═9b8302d6-8f48-4695-a901-2cafacaadbfc
# ╠═24a36630-a3c4-482b-be21-ad704abb87e9
# ╠═e359a3b5-745d-4424-8ed5-80e948cca4e4
# ╠═20c1dd23-7445-4c79-93b1-e985ed9f355c
# ╠═c6ffd653-9907-4695-8f1c-0edbb03b2d19
# ╠═cd17ec88-246b-43fe-8975-9790c2bc2d10
# ╠═dad5c54d-e361-4e47-a82e-96394fefcdce
# ╠═597c4b7b-f912-4232-9bff-f021ab3064c1
# ╠═bc1c40e9-dd17-4516-bb0b-22252fafce6b
