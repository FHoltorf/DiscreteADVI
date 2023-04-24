using CairoMakie

function plot_chain(chain; burnin = 1000)
    fig = Figure(fontsize=28)
    ax_α = Axis(fig[1,1], xlabel = L"\alpha_1", ylabel=L"\alpha_2")
    scatter!(ax_α, chain.value[burnin:end, chain.value.axes[2][1]].data[:], 
                   chain.value[burnin:end, chain.value.axes[2][2]].data[:], 
                   color = (:black,0.5))
    ax_β = Axis(fig[1,2], xlabel = L"\beta_1", ylabel=L"\beta_2")
    scatter!(ax_β, chain.value[burnin:end, chain.value.axes[2][3]].data[:], 
                   chain.value[burnin:end, chain.value.axes[2][4]].data[:], 
                   color = (:black,0.5))
    fig
end

function plot_gaussian(Σ, μ, width)
    U, S, _ = svd(Σ)
    Σ_inv = inv(Σ)
    x1_range = range(-sqrt(S[1])*width,sqrt(S[1])*width,length=100)
    x2_range = range(-sqrt(S[2])*width,sqrt(S[2])*width,length=100)
    coords_x = [μ[1] .+ U[1,:]'*[x1,x2] for x1 in x1_range, x2 in x2_range]
    coords_y = [μ[2] .+ U[2,:]'*[x1,x2] for x1 in x1_range, x2 in x2_range]
    X = range(minimum(coords_x), maximum(coords_x), length=100)
    Y = range(minimum(coords_y), maximum(coords_y), length=100)
    Z = [1/2*([x,y]-μ)'*Σ_inv*([x,y]-μ) for x in X, y in Y]
    return X,Y,Z
end

function pairplot(chain, ϕ, true_pars; burnin = 5000, width = 5, thinning = 1000)
    μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
    Σ = V*V' + 0.01*I
    symbols = [Symbol("αβ[$i]") for i in 1:4]
    vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
    chain_mean = [mean(chain[par][burnin:end]) for par in symbols]

    fig = Figure(fontsize=28, resolution = (1000,1000))

    axs = [[Axis(fig[i,j-1], xlabel = vars[j], ylabel=vars[i]) for j in i+1:4] for i in 1:3]
    burnin = 25000
    for i in 1:3
        k = 0
        for j in i+1:4
            k += 1
            X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
            scatter!(axs[i][k], chain[symbols[j]][burnin:end].data[1:thinning:end], 
                                chain[symbols[i]][burnin:end].data[1:thinning:end], 
                                color = (:black,0.2))
            global cf = contour!(axs[i][k], X, Y, -Z, linewidth = 2, 
                                colorrange = (-width, 0), 
                                colormap=:OrRd, levels= -width:1:-1)
            scatter!(axs[i][k], [chain_mean[j]], [chain_mean[i]], 
                                color = :white, marker = :cross, strokecolor=:black, strokewidth=1, markersize=15)
            scatter!(axs[i][k], [μ[j]], [μ[i]], marker = :circle, color = :red, markersize=15)
            scatter!(axs[i][k], [true_pars[j]], [true_pars[i]], marker = :star4, color = :magenta, markersize=15)
        end
    end
    legends = [[MarkerElement(marker=:circle, color = :red, markersize=15),
                MarkerElement(marker=:cross, color = :white, strokecolor=:black, strokewidth=1, markersize=15),
                MarkerElement(marker=:star4, color = :magenta, markersize=15),
                MarkerElement(marker=:circle, color = (:black, 0.2), markersize=15)]]
    labels = [[L"\text{VI mean}", 
              L"\text{HMC mean}", 
              L"\text{True value}",
              L"\text{HMC samples}"]]
    ga = fig[2:3, 1] = GridLayout()
    Legend(ga[1,1], legends, labels, ["Legend"])
    Colorbar(ga[2,1], cf, #ticks=(collect(-width:1:0), ["$i" for i in abs.(-width:1:0)]),
                      label = L"\propto \log \, q", width=30)
    #colgap!(ga, 120)
    Label(ga[2,1,Top()], "Variational\nDistribution", font = :bold, padding = (0, 0, 10, 0))
    fig
end

# plot a movie 
function pairplot_movie(chain, ϕ_trace, true_pars; filename = "VI_movie.mp4", burnin = 5000, width = 5, thinning = 1000)
    ϕ = ϕ_trace[end]
    obs = Dict()

    μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
    Σ = V*V' + 0.01*I
    symbols = [Symbol("αβ[$i]") for i in 1:4]
    vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
    chain_mean = [mean(chain[par][burnin:end]) for par in symbols]

    fig = Figure(fontsize=28, resolution = (1000,1000))

    axs = [[Axis(fig[i,j-1], xlabel = vars[j], ylabel=vars[i]) for j in i+1:4] for i in 1:3]
    burnin = 25000
    for i in 1:3
        k = 0
        for j in i+1:4
            k += 1
            scatter!(axs[i][k], chain[symbols[j]][burnin:end].data[1:thinning:end], 
                                chain[symbols[i]][burnin:end].data[1:thinning:end], 
                                color = (:black,0.2))
            scatter!(axs[i][k], [chain_mean[j]], [chain_mean[i]], 
                                color = :white, strokecolor=:black, strokewidth=1, marker = :cross, markersize=15)
            scatter!(axs[i][k], [true_pars[j]], [true_pars[i]], marker = :star4, color = :magenta, markersize=15)

            X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
            obs[i,j] = Observable(X), Observable(Y), Observable(-Z), Observable(Point2(μ[j], μ[i]))
                                
            global cf = contour!(axs[i][k], obs[i,j][1], obs[i,j][2], obs[i,j][3], linewidth = 2, 
                                 colorrange = (-width, 0), 
                                 colormap=:OrRd, levels= -width:1:-1)
            
            scatter!(axs[i][k], obs[i,j][4], marker = :circle, color = :red, markersize=15)
        end
    end
    legends = [[MarkerElement(marker=:circle, color = :red, markersize=15),
                MarkerElement(marker=:cross, color = :white, strokecolor=:black, strokewidth=1, markersize=15),
                MarkerElement(marker=:star4, color = :magenta, markersize=15),
                MarkerElement(marker=:circle, color = (:black, 0.2), markersize=15)]]
    labels = [[L"\text{VI mean}", 
              L"\text{HMC mean}", 
              L"\text{True value}",
              L"\text{HMC samples}"]]
    ga = fig[2:3, 1] = GridLayout()
    Legend(ga[1,1], legends, labels, ["Legend"])
    Colorbar(ga[2,1], cf, #ticks=(collect(-width:1:0), ["$i" for i in abs.(-width:1:0)]),
                      label = L"\propto \log \, q", width=30)
    Label(ga[2,1,Top()], "Variational Distribution", font = :bold, padding = (0, 0, 10, 0))
    #it_label = Observable(1)
    #gb = fig[3, 2] = GridLayout()
    #it = lift(i -> string("Iteration = ", it_label.val), 1)
    #Label(gb[1,1], it)#, font = :bold)
    
    record(fig, "../figures/"*filename, enumerate(ϕ_trace)) do (iter,ϕ)
        μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
        Σ = V*V' + 0.01*I
        for i in 1:3
            for j in i+1:4
                X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
                obs[i,j][1][] = X
                obs[i,j][2][] = Y
                obs[i,j][3][] = -Z
                obs[i,j][4][] = Point2(μ[j], μ[i])
            end
        end
        #it_label[] = iter
        #it_label = it_label[]
    end
end


# plot convergence
#=
batch_sizes = [1, 10, 50]
n_iterations = 100
step_size = 0.05
n_sample = [2000, 200, 10]

results = Dict()
for (i,batch_size) in enumerate(batch_sizes)
    sample_traces = zeros(0, n_iterations)
    for n in 1:n_sample[i]
        opt_trace, ϕ_opt = optimize_elbo(batch_size, n_iterations, step_size)
        sample_traces = vcat(sample_traces, opt_trace')
    end
    results[batch_size] = Dict(:mean => [mean(r) for r in eachcol(sample_traces)],
                               :std => [std(r) for r in eachcol(sample_traces)])
end

function plot_opt_trajectories(batch_sizes, traces)
    fig = Figure(fontsize=18)
    ax = Axis(fig[1,1], xlabel = "Iterations", ylabel = "ELBO", xscale=log2)
    for batch_size in batch_sizes
        lines!(ax, traces[batch_size], label = "batch size = $batch_size", linewidth=2)
    end
    axislegend(position=:rb)
    fig
end

function plot_sgd_traces(results; colors = [:red, :magenta, :purple, :magenta, :orange],
                                  alpha = 0.4)
    fig = Figure(fontsize=28)
    ax = Axis(fig[1,1], xlabel = "Iteration", ylabel = "ELBO")
    k = 0
    for batch_size in keys(results)
        k += 1
        band!(ax, 1:length(results[batch_size][:mean]),
                  results[batch_size][:mean] .- results[batch_size][:std],
                  results[batch_size][:mean] .+ results[batch_size][:std], 
                  color = (colors[k], alpha))
        lines!(ax, results[batch_size][:mean], 
                   color = colors[k], label = "sample size = $batch_size", 
                   linewidth=4)
    end
    axislegend(position = :rb)
    fig
end
=#