# Universal Adversarial Triggers
- find a phrase that if inserted in an input would cause prediction $y$
- generally another method for the same outcome as [[Anchors]]
- it does not necessarily matter where to put the triggers
- triggers are being used on the dataset globally

## Generating Triggers
- equal approach to [[Saliency Maps]], see:
![[Trigger Computation.png]]

## Examples
### SNLI Hypothesis

- as soon as a model sees word nobody; the answer is **contradiction**
- due to data artifacts token 'nobody' had a high correlation with label contradiction
- finding triggers and data artifacts, such that a bias might be discovered

### SQuAD

- group questions by w-questions
- insert the trigger into input with the hypothesis to trigger a certain kind of answer
- triggers reveal model biases w.r.t question types
	- lexical overlap
	- local context bias

## Ideas
See: [[Scratchpad#Not yet explored]]

## Resources
> [[Wallace et al. 2019]] 
> [[Song et al. 2020]] 
> [[Atanasova et al. 2020]]
> [[Song et al. 2020]]
> [[Bowmann et al. 2015]]
> [[Gururangan et al. 2018]] 
> [[Poliak et al. 2018]]
> [[Rajpurkar et al. 2016]]
> [[Sugawara et al. 2018]]
> [[Jia and Liang 2017]]