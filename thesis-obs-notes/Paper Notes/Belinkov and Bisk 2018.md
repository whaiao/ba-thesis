> [[Belinkov and Bisk 2018.pdf]]
- Objective: Building a model that is resilient to noise using [[Adversarial Attacks]], while char-based NMT systems can capture morphological languages well and keep embedding dimension sizes low, they suffer from typos
- evaluations on German, French and Czech from English
- all models suffer a significant drop in [[BLEU]] when evaluated on noisy texts
- NMT based on character-based embeddings are more robust to noise
- noise types:
	- natural:
		- typos
		- misspellings
	- artifical:
		- swapping two characters
		- randomizing middle characters
		- fully randomizing
		- keyboard typos
- 	models tested:
	- 	char2char [[Lee et al. 2017]]
	- 	Nematus [[Sennrich et al. 2017]]
	- 	charCNN [[Belinkov and Bisk 2018]]
	
> Mentioned in: