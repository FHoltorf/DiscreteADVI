include("model_fast.jl")
include("visualizations.jl")
include("utils.jl")
N_adaptation = 1000
N_chain = 100000
include("jags_model.jl")

using RData, DataFrames, Dates, CategoricalArrays, FileIO

specs = [4]
N_samples = 50
for spec in specs
    Random.seed!(1)
    traces_ensemble = []
    times_ensemble = []
    for i in 1:N_samples
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

        # n_chain = 100000
        # # NUTS
        # sampler = NUTS(50, 0.65) #1000, 0.65)
        # t_MCMC = @elapsed chain_NUTS = sample(marginal_occupancy(om), sampler, n_chain, drop_warmup=true)
        # lp_trace_NUTS = compute_log_posterior_trace(chain_NUTS, chain_NUTS.name_map[1], om, :marginal)

        # JAGS
        W_mat = zeros(n, 6)
        Y_mat = zeros(n, 6)
        for i in 1:length(W)
            W_mat[i,1:length(W[i])] = [w[1] for w in W[i]]
            Y_mat[i,1:length(Y[i])] = Y[i]
        end
        X_mat = [x[1] for x in X]

        data = Dict(
                "x" => X_mat,
                "w" => W_mat,
                "y" => Y_mat,
                "n" => n,
                "alpha_mean" => [0.0,0.0],
                "alpha_prec" => 1e-3*Matrix(I,2,2),
                "beta_mean" => [0.0,0.0],
                "beta_prec" => 1e-3*Matrix(I,2,2),
                "K" => K,
                )

        ϕ0 = randn(20+n-sum(z_known))        
        inits = [
            Dict("alpha" => ϕ0[1:2], "beta" => ϕ0[3:4], "z" => Float64[z_known[i] == 1 ? 1 : rand(0:1) for i in eachindex(z_known)]),
            ]

        t_jags = @elapsed chain_JAGS = jags(jagsmodel, data, inits, ProjDir)
        jags_parnames = [Symbol("alpha[1]"), Symbol("alpha[2]"), Symbol("beta[1]"), Symbol("beta[2]")]
        true_mean = load("results/means_$(spec).jld2")["true_means"]
        trace_JAGS = compute_mean_trace(chain_JAGS, [Symbol("alpha[1]"), Symbol("alpha[2]"), 
                                                        Symbol("beta[1]"), Symbol("beta[2]")], true_mean) 
        
        n_chain = N_chain
        sampler = NUTS(N_adaptation, 0.65) 
        t_NUTS = @elapsed chain_NUTS = sample(marginal_occupancy(om), sampler, n_chain, drop_warmup=false, init_theta = ϕ0[1:4])
        trace_NUTS = compute_mean_trace(chain_NUTS, chain_NUTS.name_map[1], true_mean)                                             
        # variational inference
        #vcat(zeros(4), reshape(Matrix(I,4,4), 16), zeros(20+n-sum(z_known))) #randn(20+n-sum(z_known)) 
        # 5, 0.05, 500 is good but slow
        # 1, 0.01, 500 is bad and relatively slow
        # 5, 0.1, 200
        ϕ_opt, ϕ_trace, VI_times, elbo_trace, log_predictive_trace = optimize_elbo(om, ϕ0, 1, 1000, 0.05, n_switch = 300, n_snapshots=100000, n_estimator = 1000, marginal = false)

        traces = [[norm(ϕ[1:4] - true_mean)/norm(true_mean) for ϕ in ϕ_trace], 
                [norm(row)/norm(true_mean) for row in eachrow(trace_JAGS)],
                [norm(row)/norm(true_mean) for row in eachrow(trace_NUTS)]]
        times = [range(0, VI_times[end], length=length(traces[1])+1)[2:end], 
                range(0, t_jags,length=size(trace_JAGS,1)+1)[2:end],
                range(0, t_NUTS, length=size(trace_NUTS,1)+1)[2:end]]
        push!(traces_ensemble, traces)
        push!(times_ensemble, times)
    end
    mean_traces = [mean([traces[k] for traces in traces_ensemble]) for k in 1:3]
    std_traces = [std([traces[k] for traces in traces_ensemble]) for k in 1:3]
    spreads = [[minimum(reduce(hcat, traces[k] for traces in traces_ensemble), dims=2)[:],
                maximum(reduce(hcat, traces[k] for traces in traces_ensemble), dims=2)[:]] for k in 1:3]
    quants = [quantile([traces[k] for traces in traces_ensemble],[0.05,0.95]) for k in 1:3]            
    mean_times = [mean([times[k] for times in times_ensemble]) for k in 1:3]
    names = ["VI", "JAGS", "NUTS + marginalization"]
    

    # visualizations
    fig = trace_plot_with_spreads(mean_times, mean_traces, spreads, names, ylabel = L"\f    rac{\Vert \mu_{\text{estimated}} - \mu_{\text{true}} \Vert_2}{\Vert \mu_{\text{true}} \Vert_2}", colors = [:black, :red, :blue])
    save("../figures/mean_error_$(spec).pdf", fig)
    display(fig)
end

