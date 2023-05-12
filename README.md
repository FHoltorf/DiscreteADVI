# DiscreteADVI
6.7830 final project

This repository contains the code for the project titled "Automating Variational Inference with Discrete Random Variables".
The aim of this project is to gauge to which extent [StochasticAD.jl](https://github.com/gaurav-arya/StochasticAD.jl) can enable an ADVI-like[1] experience for probabilistic models with discrete latent features.
To that end, we study variational inference for a simply occupancy model from the population ecology literature [2]. 
Specifically, we adapt use the data by Clark et al. (2016) [3] and adapt their case study concerned with the occupancy of different
spatial territories in southern Africa by 5 different bird species. The aim is to provide a posterior summary of the 
proportion of occupied sites as well as the statistics of logistic regression coefficients relating the detection and occupancy probability to known covariates.

Throughout the repository the bird species are numbered 1-5. The assignment is as follows:
1 - Black-headed heron
2 - Egyptian goose
3 - Longclaw
4 - Sparrow-Weave 
5 - Widowbird


[1] Kucukelbir, Alp, et al. "Automatic variational inference in Stan." Advances in neural information processing systems 28 (2015).
[2] MacKenzie, Darryl I., et al. "Estimating site occupancy rates when detection probabilities are less than one." Ecology 83.8 (2002): 2248-2255.
[2] Clark, Allan E., Res Altwegg, and John T. Ormerod. "A variational Bayes approach to the analysis of occupancy models." PloS one 11.2 (2016): e0148966.