# Input Reduction
- what if we remove the **least** important words, but keep the model prediction the **same**

## Example
1. What did Tesla ~~spend~~ Astor's money on? ==0.78==
2. What did Tesla Astor's ~~money~~ on?	==0.74==
3. What did Tesla Astor's ~~on~~? ==0.76==
4. ~~What~~ did Tesla Astor's? ==0.80==
5. did ~~Tesla~~ Astor's? ==0.82==
6. did ~~Astor's~~? ==0.89==
7. did ==0.91==

**did** is the reduced word, where confidence increased from 0.78 -> 0.91.

## Problems
- input gets heavily modified
- we often do not understand results (and how to improve this)
- search is hard; gradient jumps around (things that have not been important is now being important)
	- too many reduced inputs exist
- find the nearest example which produces a different prediction to prevent heavy modification
-> [[Adversarial perturbations]]

- to produce short sentences we need to use [[Beam search]].
> [[Feng et al. 2018]]