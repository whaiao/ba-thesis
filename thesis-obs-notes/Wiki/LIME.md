# LIME
- look at model's predictions for similar inputs, while closer points are more important than further.
- fit a linear model, its weights are feature importances
## Problems
- Customizing perturbations and distances is difficult to define intuitively
- difficult to understand
	- different explanations with different perturbation
- how do we define the distance between sentences?
- expensive; underlying model needs to run inference often
- difficult to apply to non-classification tasks

> [[Ribeiro et al. 2016]]


