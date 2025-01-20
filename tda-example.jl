### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6841d6e0-87f8-11ef-20a8-c334e30900d0
begin
	using Pkg
	Pkg.add("Ripserer")
	using Ripserer
	using Plots
	using Random
	Pkg.add("PlutoUI")
	using PlutoUI
	Pkg.add("LazySets")
	using LazySets
end

# ╔═╡ a3813677-b7bb-4cdb-a227-a7b6e5e62d77
X = [
	(0.0, -1.0),
	(1.0, 0.0),
	(0.0, 1.0),
	(-1.0, 0.0),
	(0.5, 0.5),
	(-0.5, 0.5),
	(1.5, -1.5)
]

# ╔═╡ ac72bfbe-bedf-46f5-875b-d9106814bc47
dgm = ripserer(Alpha(X))

# ╔═╡ c1a1e9df-676e-46dc-8e9b-12bb8f5a7be3
@bind α Slider(0:0.01:2.0, show_value=true)

# ╔═╡ 8a27164a-6d51-4791-8071-3929699eaac6
Balls = [Ball2([x...], α/2) for x in X]

# ╔═╡ a4528277-c2ea-4ffd-94c7-c3a44dfa5e88
begin
	plt = plot(Balls, c=:orange, lw=0, fa=1.0, lim=(-2, 2))
	plt = scatter!(X, c=:black, ratio=1, label="")
end

# ╔═╡ 2faf93f2-d45d-46fc-ad73-29f05f2a567a
begin
	dgmplot = plot(dgm, grid=false, markersize=2, markeralpha=1)
	vline!([α], ls=:dash, lw=0.5, c=:black, label="")
	hline!([α], ls=:dash, lw=0.5, c=:black, label="")
end

# ╔═╡ cddf1a22-4646-47c9-aa39-4ac88053b00a
plot(
	plt, dgmplot, layout=(1, 2), size=(350, 350) .* (2, 1)
)

# ╔═╡ 875f7651-3715-460d-91c8-663cc076f35a
md"""
***

This is markdown and $\TeX$

$\alpha = \beta + \gamma$
"""

# ╔═╡ 0b7a8c2a-a204-40fc-ac65-49fbb971ac06
begin
	coef = [0; -2; 2; 1; -2; 2]
	f(x) = sum(x .^ [0:1:5...] .* coef)
end

# ╔═╡ 2760935e-6985-4a81-a82d-6ca6a1b925e6
xseq = [-1.2:0.01:1.2...]

# ╔═╡ 5fce7112-200e-4fa5-99f5-c533840c510f
plt1 = plot(xseq, x -> f(x), label="f")

# ╔═╡ 8a057b0c-e398-419c-b7e1-6bd03ac09245
@bind t Slider(-1:0.01:3)

# ╔═╡ 8da75478-01d7-4598-b453-711a47e73db3
Levelset = f.(xseq) .< t

# ╔═╡ 9d39b627-989f-431e-9796-28793ec20714
begin
	scatter(plt1, xseq, -2 .* ones(length(Levelset)), group=Int.(Levelset), label="", msw=0)
	hline!([t], c=:black, ls=:dash, label="")
end

# ╔═╡ 6aa22551-1505-47b7-a4a6-64dc19dcd743
md"---"

# ╔═╡ 6e89db05-9c8d-4381-8e95-a3c01d565296
begin
	x = range(0, 1.5, length=1000)
	f1(x) = 0.5 * (1 .- (1 .+ 0.5 .* sin.(5 .* x)) .* sin.((8 * pi * x) ./ 2)) .+ 0.5
	y = f1.(x)
	p11 = plot(x, y, label="", size=(400,400))
end

# ╔═╡ fea9bcd4-3db7-4576-92d5-d234aac9f7e6
begin
	dgm1 = ripserer(Cubical(y), dim_max=1)
	p22 = plot(dgm1, label="", size=(400,400), title="")
end

# ╔═╡ a1bd0bf8-b110-4b33-ae55-6e34927b96e1
@bind l Slider(0:0.01:2.0, show_value=true)

# ╔═╡ e67e825a-8295-4771-918e-14d0fef9df44
begin
	indx = y .> l
	x_active = x[indx]
	y_active = y[indx]
	p12 = plot(x, y, label="", size=(400,400), lw=2, c=:red)
	p12 = plot(p12, x_active, y_active, label="", lw=2, c=:black)
	hline!(p12, [l], label="level=$l")
	p23 = plot(dgm1, label="", size=(400,400), title="")
	vline!([l], ls=:dash, lw=0.5, c=:black, label="")
	hline!([l], ls=:dash, lw=0.5, c=:black, label="")
	plot(p12, p23, layout=(1,2), size=(800, 400))
end

# ╔═╡ 9101bd11-aa21-409d-b2a0-8f2460268217
md"""
1. Generate $n=100$ random points from a circle in $R^2$
2. Compute the persistence diagram from Rips/Cech/Alpha complex
3. Construct a kernel density estimator for the points
4. Construct a Cubical persistence diagram of the kernel density estimator
"""

# ╔═╡ Cell order:
# ╠═6841d6e0-87f8-11ef-20a8-c334e30900d0
# ╠═a3813677-b7bb-4cdb-a227-a7b6e5e62d77
# ╠═8a27164a-6d51-4791-8071-3929699eaac6
# ╠═a4528277-c2ea-4ffd-94c7-c3a44dfa5e88
# ╠═ac72bfbe-bedf-46f5-875b-d9106814bc47
# ╠═2faf93f2-d45d-46fc-ad73-29f05f2a567a
# ╠═c1a1e9df-676e-46dc-8e9b-12bb8f5a7be3
# ╟─cddf1a22-4646-47c9-aa39-4ac88053b00a
# ╠═875f7651-3715-460d-91c8-663cc076f35a
# ╠═0b7a8c2a-a204-40fc-ac65-49fbb971ac06
# ╠═5fce7112-200e-4fa5-99f5-c533840c510f
# ╠═2760935e-6985-4a81-a82d-6ca6a1b925e6
# ╠═8da75478-01d7-4598-b453-711a47e73db3
# ╠═8a057b0c-e398-419c-b7e1-6bd03ac09245
# ╠═9d39b627-989f-431e-9796-28793ec20714
# ╟─6aa22551-1505-47b7-a4a6-64dc19dcd743
# ╠═6e89db05-9c8d-4381-8e95-a3c01d565296
# ╠═fea9bcd4-3db7-4576-92d5-d234aac9f7e6
# ╟─a1bd0bf8-b110-4b33-ae55-6e34927b96e1
# ╠═e67e825a-8295-4771-918e-14d0fef9df44
# ╟─9101bd11-aa21-409d-b2a0-8f2460268217
