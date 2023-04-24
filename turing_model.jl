using Turing

@model function occupancy(Y;z_known = z_known, prior = αβ_prior, W=W, X=X)
    αβ ~ prior
    Turing.@addlogprob! logmarginal(Y,z_known,αβ,W=W,X=X)
end    
