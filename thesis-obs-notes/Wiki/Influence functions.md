# Influence functions 
- **Goal**: for a given test prediction, identify the most influential training
- consider $x$ a test point and $z$ a training point, s.t. $I(x,z) =$ what is the influence of $z$ for the prediction for $x$.
- remove $z$ to see how parameters change -> quantize & capture change in parameters
- works empirically for any models
- scales poorly with model size and data 
- can only be applied in convex settings, thus might struggle in neural approaches 
- alternative: [[Representer point selection]]

## How to Compute
1. Pick $\hat{\theta}$ to minimize $\frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(z_i, \theta)$, so basically train your classifier, then leave a training sample away to get ->
2. $\hat{\theta}_{-z_{train}}$ to minimize: $\frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i, \theta) - \frac{1}{n}\mathcal{L}(z_{train}, \theta)$
3. look at the loss difference: $\mathcal{L}(z_{test},  \hat{\theta}_{-z_{train}}) - \mathcal{L}(z_{test}, \hat{\theta})$
	**Computation of: $\hat{\theta}_{-z_{train}}$ is horrendous, hence classifier must be retrained again and again**, (we need to approximate)

> [[Koh and Liang 2017]]
> [[Basu et al. 2020]]