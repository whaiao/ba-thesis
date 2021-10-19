# EMNLP Interpretability Tutorial
Date: 2021-10-19
Note: #tutorial
Topic: #interpretability #explainability #deeplearning

## Content

### Interpretability Overview
#### Notable Failures
- fragile results
- validation metrics do not capture the full story of the model
- looking at metrics is not enough

#### What else than metrics?
- [[masked language models]] and changing pronouns to see which kind of predictions are avail, while we can also see if there is any obvious bias
- misprediction due to faulty data
- we want to understand what's happening underneath and which part of the model does not behave as desired
- also to build trust, we need to increase explainability of these
- *hypothesis: if we understand the model better, we might as well gain understanding on how we can improve those*
- if we figure out what kind of patterns the model sees, we are able to learn more about perception, or perhaps about language

##### But how?
==(first three methods - acl 2020)==
- probing internal representations; like weights, what's happening, looking at neuron activations, decode from different areas of internal representations and use it to classify e.g. a POS tag
- challenge the network via challenge sets (carefully designed inputs) to see how the model reacts to it
- change the model structure itself, so it gains interpretability. This might be done via additionally letting the model create a rational about the prediction. It's not necessarily causal. Another option is to bottleneck the computation of models, such that they become easier in nature.
See: [[Deyoung et al. 2020]], [[Narang et al.2020]], [[Lei et al. 2017]], [[Subramanian et al. 2020]], [[Gupta et al. 2020]], [[Andreas et al. 2016]] 
- looking into the inputs to see what makes the prediction happening. Go for a causal analysis as looking at the input features which we can remove to make the prediction happen.
See: [[Ribeiro et al. 2016]], [[Feng et al. 2018]], [[Wallace et al. 2019]] 
- look for global decision rules which the model is running on, i.e. contradictions which the model predicts to 100%
- looking at training data, if there is any bias inherent

##### Why focus on the latter three?
- we can frame fine-grained questions on why the model fail on this particular input, or what the impact is of a particular input
- the methods are model-agnostic, fast and easy to compute, while the gradient of the loss can help on how the model learns


### What Parts of an Input Led to a Prediction?

#### [[Saliency Maps]]
- [[Saliency Maps#Via Input Gradients]]
- [[Saliency Maps#Via Pertubations]]

#### [[Perturbations]]
- [[Perturbations#Input reduction]]
- [[Adversarial perturbations]]

### What Decision Rules Led to a Prediction?
- Input interpretations are local 
- *if pattern x holds, model typically makes prediction y*
- rather find small rules which are not just based on the a large part of the input

#### [[Anchors]]
#### [[Universal Adversarial Triggers]]

- can identify global bugs (i.e. annotation artifacts)
- hard to find broad rules, s.t. we might only find highly specific rules which are not clear what action to take

See: [[Lakkaraju et al. 2017]] [[Guidotti et al. 2018]] [[Tan et al. 2018]] [[Sushil et al. 2018]] [[Ribeiro et al. 2018]] [[Li et al. 2019]]

### Which Training Examples Caused a Prediction?

- instead of just explaining input, we look from the point of view of the model to try to explain why a model predicted something given input $x$
- use [[Influence functions]] to see which training examples where the most influential for the prediction
- shows *where* the model found patterns, such that we can take actions based on what was found
- can be uninterpretable, computationally expensive
- requires approximations
- influential points too specific to choice of pretrained models?
- ==very promising area==

- See: [[Yeh et al. 2018]] [[Han et al. 2020]] [[Koh and Liang 2017]] [[Garima et al. 2020]] [[Basu et al. 2020]]

### Implementation
[[Video] Implementation starts from here](https://youtu.be/gprIzglUW1s?t=10738)

### Q&A
- Attention vs [[Saliency Maps]], (Gradient based methods vs attention based methods, since attention is also creating a [[Saliency Maps]]) see: [[Attention is not Explanation]] & [[Attention is not not Explanation]]
- Why trust a model which generates a rational? -> Learn more about the context, but more likely no causality. Some works have been using rationals which have been created internally as an additional input into the next layer.
- how to test high bias -> put specific sentences in another testing dataset
- how to distinguish triggers vs effective features?

## References

[[021 Interpretability & Explainability]]