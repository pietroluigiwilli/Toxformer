# Toxformer

## Transformer enabled toxicity prediction based on chemical structure
## UZH

## 1.	Group Members<br>
Claudia Lourenço Rodrigues. Faculty of Science.<br>
Ekaterina Maevskaia. Faculty of Science.<br>
Pietro Willi. Department of Chemistry.<br>
Erik Schiess. Department of Economics.<br>

## 1.	General Topic<br>
Before any clinical trials are performed any substance intended to become approved as an Active Pharmaceutical Ingredient needs to undergo extensive toxicity screenings during the preclinical development. The screenings are performed on cells and then on animals. Animal testing in Europe is typically exposed to strict regulations due to ethical considerations.[1] This turns the screenings into an expensive and lengthy procedure. In silico prediction of the toxicity reduces the amount of potentially dangerous substances and lowers the necessity of in vivo experiments and therefore shortens the time and cost of the study.

The toxicity of a substance depends on the extent of its ability to interfere with biological processes.[2] It can do so by various mechanisms, these can be biological, chemical, physical, radioactive or behavioural. Toxicity is measured indirectly through its effect. However the effect of toxicity is different depending on the target species, species individual, and substance administration. As such toxicity data is usually recorded as the average mass of substance per unit mass of target organism required to kill half of the tested population of target organisms. This metric is known as median lethal dose, LD50. 

The prediction of toxicity from molecular descriptors is an attractive area of research for data science. Numerous papers illustrate attempts at predicting toxicity. The majority of them use a machine learning approach. More recent papers have mainly focused on the use of deep learning. Chen et al. used a graph convolutional architecture to encode a description of the molecular structure and predict the toxicity of molecules from various datasets including the Tox21 dataset [3].  Zhang et al. explored the use of various algorithms, deep learning and decision tree based, to predict toxicity of molecules from the Tox21 challenge data. They found the deep learning algorithms, in particular graph neural networks to work best [4].  Recently, Cremer et al. used a graph transformer and 3D representation of molecules as graphs 
