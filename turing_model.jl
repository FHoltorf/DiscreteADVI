using Turing

@model function occupancy(prior,Y,z_known,W,X)
    αβ ~ prior
    Turing.@addlogprob! logmarginal(Y,z_known,αβ,W=W,X=X)
end    
