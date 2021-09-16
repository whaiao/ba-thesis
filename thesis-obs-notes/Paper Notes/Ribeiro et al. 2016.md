> [[Ribeiro Et Al. 2016.pdf]]

- trust is key for users using the model at hand
- inspects individual predictions and their explanations (additionally to metrics)
- providing explanations can increase the acceptance of recommendations, see: [[Dzindolet et al 2013]]
- example results:
	- [[Data leakage]]
	- [[Dataset shift]]
- often occurring mismatch between quantitative and qualitative metrics
# Explainers
- should be **interpretable**, provide qualitative understanding between input and response (should be interpretable by the user)
- should have **local fidelity**, must behave in the vicinity of the prediction, while local fidelity does not imply global fidelity
- should be **model agnostic**, treat the model as a black box
- should have a **global perspective** to assert trust

# LIME – Local Interpretable Model-agnostic Explanations
- possible interpretation representation in form of one-hot encoded vector showing if an input is present, or [[Super-Pixel]]
$$LIME = \xi(x) = \underset{x}{\arg\min} \mathcal{L}(f, g,\pi_x) + \Omega(g)$$
given:
$g \in G \coloneqq$ explanation families
$g \in {0, 1}^{d^{'}}$, domain of model
$\Omega(g)$, measure of complexity (i.e. depth in a decision tree)
$f \colon \mathbb{R}^d \rightarrow \mathbb{R}, f(x)$, probability that $x$ belongs to a certain class
$\mathcal{L}(f,g,\pi_x)$, measure of how unfaithful $g$ is in approximating $f$ in the locality defined by $\pi_x$
$z\in \mathbb{R}^d, f(x) \text{is used as a label for }G$
$z' \in \mathcal{Z}$, [[Perturbations]] sample from $x'$

![[LIME Example.png]]

> Mentioned in:
> [[EMNLP Interpretability Tutorial]]