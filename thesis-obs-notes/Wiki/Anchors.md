# Anchors
- finding a global if-then rule on which the model predicts a certain pattern
	- POS-tag: example: if previous word is a particle the next one is a verb
	
- find a perturbation distribution which are similar to the input, but must not be syntactically or semantically the same
- treat it as an optimization problem, trying to optimize the prediction on the anchors $\mathcal{A}$. Whereas $D_x(\bullet | \mathcal{A})$, is the distribution of perturbations, with which the model should learn high coverage and learning to predict the same class as beforehand
- anchors need a local starting point to search for valid anchors

## How to Compute
1. Consider single element rules; find rules which might be anchors
2. Sample from $D_x(\bullet | \mathcal{A})$, run model and estimate precision
3. Choose highest precision rule, might as well investigate more element rules
4. if precision threshold is reached return anchor


>  [[Ribeiro et al. 2018]]