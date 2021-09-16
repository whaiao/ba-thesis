# Saliency Maps
- compute the relative importance of each token in the input
- importance of: if you change or remove the token, how much is the prediction affected? i.e. masking in language models (on what is the prediction on the mask conditioned)

## Generation
### Via Input Gradients
**TLDR**
- i.e. with a "tiny change" to the feature, what happens to the prediction? Direction of gradient is essential to see how we might look at models
- importance values are then visualized in a [[Heatmap]]
- needs access to the model
- not customizable
	- small changes in a token might not be meaningful
	- distance is $L_2$
- gradients can be unintuitive
- difficult to apply in other tasks than classification
- estimate the importance of a feature using derivative of output w.r.t that feature

**Bold** statements are recommended by the speakers.
#### What to use with an output?
- top predictions probability
- top prediction [[Logits]]
- **Loss (with the top prediction as the ground-truth class)**
#### What happens with multiple outputs?
- text generation
- tagging

#### What to use with an input token?
Input is usually an embedding, such that:
- sum it
- take an $L_p$ norm
- ** dot product with embeddings itself (since it already has a value with it?)**

#### What do show to the user?
- how is the information displayed?

#### How to compute?
$-\nabla_{e(t)}\mathcal{L}_{\hat{y}}\cdot e(t)$
Gradient with respect to the loss, taking the dot product with the embedding token.

#### Problems
- gradient too local and thus sensitive
- saturated outputs (after activation functions, or softmax) -> unintuitive gradients
- discontinuous gradients e.g. thresholding

#### How to mitigate?
- TL;DR: don't rely on a single gradient calculation
- [[SmoothGrad]], add [[Gaussian noise]] to input and average the gradient
- [[Integrated Gradient]], average gradients along path from zero to input
- other: [[LRP]], [[DeepLIFT]], [[GradCAM]]

#### Resources

[[Ribeiro et al. 2016]] [[Murdoch et al. 2018]] [[Wallace et al. 2019]] [[Han et al. 2020]] [[Simonyan et al. 2014]] [[Shrikumar et al. 2017]] [[Smilkov et al. 2017]] [[Sundararajan et al. 2017]]

***
### Via [[Perturbations]]

**TLDR**
- black-box; model-agnostic
- allow input perturbations/neighborhoods, use different units of explanations
	- words, phrases sentences
	- multimodality
	- what perturbations are valid?
	
#### Methods
- [[Leave-one-out]]
- [[LIME]]

#### Resources

[[Li et al. 2017]]

## Problems
- linear representations can be limited

