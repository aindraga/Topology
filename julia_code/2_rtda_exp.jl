import RobustTDA as rtda
using RobustTDA: DistanceFunction
using BenchmarkTools
using NearestNeighbors

# Dataset Generation
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

function my_dtm(
    data::AbstractVector{T},
    m::Real
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    tree = KDTree(reduce(hcat, data), leafsize = 1)

    return DistanceFunction(
        k = floor(Int, m * length(data)),
        trees = [tree],
        X = [data],
        type = "dtm",
        Q = 1
    )
end

function my_dist(
    data::AbstractVector{T}
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    Xq = [data]

    trees = [BruteTree(reduce(hcat, xq), leafsize = 1) for xq in Xq]

    return DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "dist",
        Q = 1
    )
end

small_data = myCircle(10, 10)
rtda.dist(small_data)
my_dist(small_data)
rtda.dtm(small_data, 0.1)
my_dtm(small_data, 0.1)


curr_data = myCircle(50_000, 50_000)

kd_nn = @btimed rtda.dist(curr_data)
brute_nn = @btimed my_dist(curr_data)

kd_dtm = @btimed my_dtm(curr_data, 20)
brute_dtm = @btimed rtda.dtm(curr_data, 20)

println("KDTree NN Function: $(kd_nn.time)")
println("BruteTree NN Function: $(brute_nn.time)")

println("KDTree DTM Function: $(kd_dtm.time)")
println("BruteTree DTM Function: $(brute_dtm.time)")
