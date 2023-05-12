include("model_fast.jl")
include("visualizations.jl")
include("utils.jl")


using RData, DataFrames, Dates, CategoricalArrays, FileIO

specs = [1,2,3,4,5]
for spec in specs
    Random.seed!(1)
    data = load("design_mats/design_mat$(spec).Rdata")
    function organize_data(data)
        X = [vcat(x, 1) for x in data["design_mats"]["X"].AET_div_PET_s]
        Y = Vector{Bool}[]
        W = Vector{Vector{Float64}}[]
        n = 0
        for k in data["design_mats"]["nvisits"]
            push!(Y, data["design_mats"]["Y"].V1[n+1:n+k])
            push!(W, [vcat(data["design_mats"]["W"].nspp[i],1) for i in n+1:n+k])
            n += k
        end
        return X, Y, W
    end

    X, Y, W = organize_data(data)

    K = [length(y) for y in Y] # visits per site
    n = length(Y) # number of sites

    # regression coeff priors
    αβ_prior = MvNormal(Float64[0,0,0,0], 1000*I)
    om = OccupancyModel(Y,Bool[],W,X,αβ_prior)

    z_known = om.z_known

    n_chain = 200000
    # NUTS
    sampler = NUTS() #1000, 0.65)
    t_NUTS = @elapsed chain_NUTS = sample(marginal_occupancy(om), sampler, n_chain, drop_warmup=true, burnin=25000)
    true_means = mean(chain_NUTS).nt[2]
    println(true_means)
    save("results/means_$(spec).jld2", "true_means", true_means)
    
end