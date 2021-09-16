
> [[Wallace et al. 2019.pdf]]
- provides an extension for AllenNLP to use for interpretability
- provides following methods to inspect interpretability:
	- gradient based saliency maps: 
		- gradient-based methods determine this importance using gradient of the loss wrt. tokens[[Simonyan et al. 2014]]
		- Vanilla Gradient visualized gradient of loss wrt. each token [[Devlin et al. 2019]][[Simonyan et al. 2014]]
		- Integrated Gradients [[Sundararajan et al. 2017]], baseline $x'$ which is input without information
		- SmoothGrad [[Smilkov et al. 2017]] gradient averaging and adding noise
	- adversarial attacks:
		- HotFlip [[Ebrahimi et al. 2018]], uses gradient to replace words to change a model's prediction -> *how would the prediction change if certain words are replaced?*
		- HotFlip to cause a specific prediction
		- Input Reduction [[Feng et al. 2018]] by removing as many words as possible, to find the one causing the highest confidence in the prediction
		
> Mentioned in:
> [[EMNLP Interpretability Tutorial]]
