include("model_fast.jl") # model specifications 
include("utils.jl") # optimization routines
include("visualizations.jl") # plotting routines

using FileIO, BenchmarkTools, ForwardDiff, LaTeXStrings

# scaling of just the gradients
n_range = [2^i for i in 1:12]
grad_results = Dict()
for n in n_range
    println(n)
    K = [4 for i in 1:n]

    # covariates 
    x_dst = Uniform(-2,2)
    w_dst = Uniform(-5,5)
    X = [vcat((rand(x_dst, 1) .- mean(x_dst)) ./ std(x_dst),1) for i in 1:n]
    W = [[vcat((rand(w_dst, 1) .- mean(w_dst)) ./ std(w_dst),1) for j in 1:K[i]] for i in eachindex(K)]

    # generate data according to probabilistic model
    α_true = [1.75, 1.35]
    β_true = [2.5,-1.85]    
    Z = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n] # true latent variables
    Y = [[rand(Bernoulli(Z[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n] # observations

    # regression coeff priors
    αβ_prior = MvNormal(Float64[0,0,0,0], 1000*I)
    
    om = OccupancyModel(Y,Z,W,X,αβ_prior)

    ϕ0 = randn(20+om.n-sum(om.z_known))
    
    elbo = ELBO(om, 1)
    obj = StochasticModel(ϕ -> -elbo(ϕ), ϕ0)
    grad_results[n, :discrete] = @benchmark stochastic_gradient($obj) samples = 100
    
    ϕ0 = ϕ0[1:20]
    elbo_marginal = ELBO_marginal(om, 1)
    obj_marginal = ϕ -> -elbo_marginal(ϕ)
    grad_results[n, :forward_mode] = @benchmark ForwardDiff.gradient($obj_marginal, $ϕ0) samples = 100
end

function get_scaling(results, offset=1)
    n_range = sort(collect(keys(results)))
    times = [mean(results[n]).time*1e-9 for n in n_range]
    logtimes = log10.(times[offset:end])
    logn = log10.(n_range[offset:end])
    a = hcat(logn, ones(length(logn)))\logtimes
    return n_range, times, a
end

fig = Figure(fontsize=24)
ax = Axis(fig[1,1],
          ylabel = "wall clock time [s]", 
          xlabel = "number of sites n",
          xminorgridvisible = true,
          yminorgridvisible = true,
          xminorticks = IntervalsBetween(9),
          yminorticks = IntervalsBetween(9),
          yticks = ([10.0^(i) for i in -6:1], [latexstring("10^{$i}") for i in -6:1]),
          xticks = ([10.0^(i) for i in 0:4], [latexstring("10^{$i}") for i in 0:4]),
          xscale = log10,
          yscale = log10)
colors = [:black, :red]
label = Dict(:discrete => "StochasticAD.jl", :forward_mode => "ForwardDiff.jl + marginalization")
k=1
for mode in [:discrete, :forward_mode]
    ns, times, a = get_scaling(Dict(n=>grad_results[n,mode] for n in n_range),6)
    logn = log10.(ns)
    scatter!(ax, ns, times, color = colors[k], markersize=20)
    lines!(ax, ns, 10 .^ (hcat(logn, ones(length(logn)))*a), color = colors[k], linewidth=2,
               label = label[mode]*" (slope = $(round(a[1],digits=2)))")
    k += 1
end
axislegend(position=:lt)
fig


save("../figures/grad_scaling.pdf",fig)

save("results/grad_scaling_results.jld2", "grad_results", grad_results)

# check the estimator variance
n_range = [10, 100, 200]
n_estim_range = [2^i for i in 0:7]
results = Dict()
results_marginal = Dict()

# random occupancy model instance
n = n_range[end]
K = [4 for i in 1:n]

# covariates 
x_dst = Uniform(-2,2)
w_dst = Uniform(-5,5)
X = [vcat((rand(x_dst, 1) .- mean(x_dst)) ./ std(x_dst),1) for i in 1:n]
W = [[vcat((rand(w_dst, 1) .- mean(w_dst)) ./ std(w_dst),1) for j in 1:K[i]] for i in eachindex(K)]

# regression coeff priors
αβ_prior = MvNormal(Float64[0,0,0,0], 1000*I)
prior = αβ_prior
# generate data according to probabilistic model
α_true = [1.75, 1.35]
β_true = [2.5,-1.85]    
Z = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n] # true latent variables
Y = [[rand(Bernoulli(Z[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n] # observations
om = OccupancyModel(Y,Z,W,X,αβ_prior)
ϕ_init = randn(20+n-sum(om.z_known))
ϕ0_marginal = ϕ_init[1:20]

for n in n_range
    println("Number of sites: $n")
    
    om = OccupancyModel(Y[1:n],Z[1:n],W[1:n],X[1:n],αβ_prior)
    ϕ0 = deepcopy(ϕ_init[1:20+n-sum(om.z_known)])
    
    elbo = ELBO(om, 1)
    obj = StochasticModel(elbo, ϕ0)
    stochastic_gradient(obj)
    for n_estim in n_estim_range
        println("MC Samples: $n_estim")
        elbo = ELBO(om, n_estim)
        obj = StochasticModel(ϕ -> -elbo(ϕ), ϕ0)
        grads = [stochastic_gradient(obj).p[1:20] for i in 1:300]
        results[n, n_estim] = (mean(grads), std(grads))
        
        elbo_marginal = ELBO_marginal(om, n_estim)
        obj_marginal = ϕ -> -elbo_marginal(ϕ)
        grads = [ForwardDiff.gradient(obj_marginal, ϕ0_marginal) for i in 1:300]
        results_marginal[n, n_estim] = (mean(grads), std(grads))
    end
end

fig = Figure(fontsize=24, resolution = (800,400))
ax = Axis(fig[1,1], 
          ylabel=L"\text{tr}\, \text{Cov}\left[\tilde{\nabla}_{\lambda}\text{ELBO}\right]", 
          xlabel=L"\text{number of MC samples}",
          yticks = ([10.0^(i) for i in 0:7], [latexstring("10^{$i}") for i in 0:7]),
          xticks = ([10.0^(i) for i in 0:2], [latexstring("10^{$i}") for i in 0:2]),
          xminorgridvisible=true,
          xminorticks=IntervalsBetween(9),
          yminorticks=IntervalsBetween(9),
          yminorgridvisible=true,
          yscale=log10, xscale=log10)
markers = [:circle, :utriangle, :rect]
for (i,n) in enumerate(n_range)
    scatterlines!(ax, n_estim_range, [sum(results[n, n_estim][2] .^ 2) for n_estim in n_estim_range], 
                      marker = markers[i], color = :black)
    scatterlines!(ax, n_estim_range, [sum(results_marginal[n, n_estim][2] .^ 2) for n_estim in n_estim_range], 
                      marker = markers[i], color =:red)
end
n_leg, mode_leg = [MarkerElement(marker=:circle, color = :black),
                   MarkerElement(marker=:utriangle, color = :black),
                   MarkerElement(marker=:rect, color = :black)],
                  [LineElement(color=:red, linewidth=2), 
                   LineElement(color=:black, linewidth=2)]

n_labels, mode_labels = ["$n" for n in n_range], ["ForwardDiff.jl\n+ marginalization", "StochasticAD.jl"]
Legend(fig[1,2], [n_leg, mode_leg], [n_labels, mode_labels], ["number of sites n", "AD mode"])
fig

save("../figures/variance_comparison.pdf",fig)


save("results/variance_results.jld2", "results", results,
                                      "results_marginal", results_marginal,
                                      "n_range", n_range,
                                      "n_estim_range", n_estim_range)
