# SocialIQa, sap2019
Date: 2021-10-19
Note: #paper 
Topic: #data #commonsense

## Content
- A dataset which enables [[Transfer Learning]] for commonsense challenges
- After events have been created from Comet2020 [[hwang2021]], via crowdsourcing 37588 context-question-answer triples have been created, where mturks have been provided an event and had to create the context, the question and each a positive and negative answer
- A set of incorrect answers have been created which require reasoning about the context. While they are stylistically similar to the correct answer, they yet are subtly wrong.
- A second set of incorrect answers have been created with swapped context -> question to ensure the model does not learn stylistic patterns.
- Human performance on the data is ~87% accuracy, while their best model ([^1]BERT-large, 340M params) yields a result of 66% accuracy
- Applying [[sequential finetuning]] on commonsense knowledge as [[0311a  Choice of Plausible Alternatives, roemmele2011]] and [[0311b Winograd Schema, levesque2011]] results in a gain of up to 5%


## Related Work

Commonsense Benchmarks: [[0311b Winograd Schema, levesque2011]], [[0311a  Choice of Plausible Alternatives, roemmele2011]], [[0311c CommonsenseQA, talmor2019]]

Commonsense Knowledge Bases: 


## References
[[031 Commonsense Knowledge]]


[^1]: [Huggingface Pretrained BERT](https:/github.com/huggingface/pytorch-pretrained-BERT)