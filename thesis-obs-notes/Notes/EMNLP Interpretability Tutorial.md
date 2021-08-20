# EMNLP Interpretability Tutorial
*** 
## Interpretability Overview
### Notable Failures
- fragile results
- validation metrics do not capture the full story of the model
- looking at metrics is not enough

### What else than metrics?
- [[masked language models]] and changing pronouns to see which kind of predictions are avail, while we can also see if there is any obvious bias
- misprediction due to faulty data
- we want to understand what's happening underneath and which part of the model does not behave as desired
- also to build trust, we need to increase explainability of these
- *hypothesis: if we understand the model better, we might as well gain understanding on how we can improve those*
- if we figure out what kind of patterns the model sees, we are able to learn more about perception, or perhaps about language

#### But how?
==(first three methods - acl 2020)==
- probing internal representations; like weights, what's happening, looking at neuron activations, decode from different areas of internal representations and use it to classify e.g. a POS tag
- challenge the network via challenge sets (carefully designed inputs) to see how the model reacts to it
- change the model structure itself, so it gains interpretability. This might be done via additionally letting the model create a rational about the prediction. It's not necessarily causal. Another option is to bottleneck the computation of models, such that they become easier in nature.
See: [[Deyoung et al. 2020]], [[Narang et al.2020]], [[Lei et al. 2017]], [[Subramanian et al. 2020]], [[Gupta et al. 2020]], [[Andreas et al. 2016]] 
- looking into the inputs to see what makes the prediction happening. Go for a causal analysis as looking at the input features which we can remove to make the prediction happen.
See: [[Ribeiro et al. 2016]], [[Feng et al. 2018]], [[Wallace et al. 2019]] 
- look for global decision rules which the model is running on, i.e. contradictions which the model predicts to 100%
- looking at training data, if there is any bias inherent

#### Why focus on the latter three?
- we can frame fine-grained questions on why the model fail on this particular input, or what the impact is of a particular input
- the methods are model-agnostic, fast and easy to compute, while the gradient of the loss can help on how the model learns


## What Parts of an Input Led to a Prediction?

### [[Saliency Maps]]
- [[Saliency Maps#Via Input Gradients]]
- [[Saliency Maps#Via Pertubations]]

### [[Perturbations]]
- [[Perturbations#Input reduction]]
- [[Adversarial perturbations]]