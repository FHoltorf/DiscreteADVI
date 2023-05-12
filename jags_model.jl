ProjDir = dirname(@__FILE__)

cd(ProjDir)
using Jags, LinearAlgebra, Distributions, Statistics
ENV["JAGS_HOME"] = "/usr/local/bin/"


line = "
model {
  for (i in 1:n) {
    z[i] ~ dbern(psi[i])
    for (j in 1:K[i]) {
      y[i,j] ~ dbern(z[i]*d[i,j]);
    }
  } 
  alpha ~ dmnorm(alpha_mean, alpha_prec)
  beta ~ dmnorm(beta_mean, beta_prec)
  for (i in 1:n) {
    psi[i] <- 0.00001 + 0.99998/(1+exp(-beta[1]*x[i] - beta[2]))
  }
  for (i in 1:n) {
    for (j in 1:K[i]) {
        d[i,j] <- 0.00001 + 0.99998/(1+exp(-alpha[1]*w[i,j] - alpha[2]))
    }
  }
}
"


monitors = Dict(
  "alpha" => true,
  "beta" => true,
  "z" => true
)

jagsmodel = Jagsmodel(
  name="occupancy",
  model=line,
  monitor=monitors,
  adapt = N_adaptation,
  ncommands=1, nchains=1,
  #deviance=true, dic=true, popt=true,
  pdir=ProjDir,
  nsamples = N_chain);
jagsmodel.command[1] = Cmd([ENV["JAGS_HOME"]*"jags", jagsmodel.command[1][2]])


println("\nJagsmodel that will be used:")
jagsmodel |> display
