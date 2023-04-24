function elbo_estimator(ϕ; n=1, y=Y, prior=prior, W=W, X=X)
    elbo = 0.0
    for _ in 1:n
        z, αβ = sample_q(ϕ)
        elbo += div(ϕ, z, αβ, y=y, prior=prior, W=W, X=X)
    end
    return elbo/n
end

function predictive_posterior_estimator(ϕ; n=1, y=Y, W=W, X=X)
    log_predictive = 0.0
    for _ in 1:n
        z, αβ = sample_q(ϕ)
        log_predictive += loglikelihood(y, z, αβ, W=W, X=X)
    end
    return log_predictive/n
end

function optimize_elbo(batch_size, iterations, LEARNING_RATE; n_estimator = batch_size, n_snapshots = 1)
    ϕ0 = randn(n+20)
    obj = StochasticModel(ϕ -> -elbo_estimator(ϕ, n= batch_size), ϕ0)
    elbo_trace = Float64[]
    log_predictive_trace = Float64[]
    wall_times = Float64[]
    ϕ_trace = Vector{Float64}[]
    optimizer = Adam(LEARNING_RATE)
    setup = Optimisers.setup(optimizer, obj)
    t = 0
    elbo = elbo_estimator(obj.p, n=n_estimator)
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    for i in 1:iterations
        t += @elapsed grad = stochastic_gradient(obj)
        if sum(isinf.(grad.p)) == 0
            t += @elapsed Optimisers.update!(setup, obj, grad)
            push!(ϕ_trace, deepcopy(obj.p))
            println("iteration $i")
        end
        if i % n_snapshots == 0
            elbo = elbo_estimator(obj.p, n=n_estimator)
            log_predictive = predictive_posterior_estimator(obj.p, n=n_estimator)
            push!(log_predictive_trace, log_predictive)
            push!(elbo_trace, elbo)
            push!(wall_times, t)
        end
    end
    elbo = elbo_estimator(obj.p, n=n_estimator)
    log_predictive = predictive_posterior_estimator(obj.p, n=n_estimator)
    push!(log_predictive_trace, log_predictive)
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    return obj.p, ϕ_trace, wall_times, elbo_trace, log_predictive_trace
end
