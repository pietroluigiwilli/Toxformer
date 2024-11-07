# Toxformer

## Transformer enabled toxicity prediction based on chemical structure
## University of Zurich, Introduction to Data Science

## 1.	Group Members<br>
Claudia Lourenço Rodrigues. Faculty of Science.<br>
Ekaterina Maevskaia. Faculty of Science.<br>
Pietro Willi. Department of Chemistry.<br>
Erik Schiess. Department of Economics.<br>

## 2.	General Topic<br>
Before any clinical trials are performed any substance intended to become approved as an Active Pharmaceutical Ingredient needs to undergo extensive toxicity screenings during the preclinical development. The screenings are performed on cells and then on animals. Animal testing in Europe is typically exposed to strict regulations due to ethical considerations.[1] This turns the screenings into an expensive and lengthy procedure. In silico prediction of the toxicity reduces the amount of potentially dangerous substances and lowers the necessity of in vivo experiments and therefore shortens the time and cost of the study. [10]

The toxicity of a substance depends on the extent of its ability to interfere with biological processes.[2] It can do so by various mechanisms, these can be biological, chemical, physical, radioactive or behavioural. Toxicity is measured indirectly through its effect. However the effect of toxicity is different depending on the target species, species individual, and substance administration. As such toxicity data is usually recorded as the average mass of substance per unit mass of target organism required to kill half of the tested population of target organisms. This metric is known as median lethal dose, LD50. 

The prediction of toxicity from molecular descriptors is an attractive area of research for data science. Numerous papers illustrate attempts at predicting toxicity. The majority of them use a machine learning approach. More recent papers have mainly focused on the use of deep learning. Chen et al. used a graph convolutional architecture to encode a description of the molecular structure and predict the toxicity of molecules from various datasets including the Tox21 dataset [3].  Zhang et al. explored the use of various algorithms, deep learning and decision tree based, to predict toxicity of molecules from the Tox21 challenge data. They found the deep learning algorithms, in particular graph neural networks to work best [4].  Recently, Cremer et al. used a graph transformer and 3D representation of molecules as graphs in order to predict toxicity [5]. The aim of this project is to use a transformer architecture to predict toxicity based on the molecular structure. The specific representation to be used for the molecular structure is to be decided . As a benchmark we will use GPT3. Jablonka et al demonstrated that GPT3 can outperform custom designed and custom trained models when trained using SMILES.[6] 
## 3.	Motivation <br>
The a priori prediction of molecular toxicity based solely on the structure would represent an important tool for drug discovery and development. Drug discovery is a very time consuming process often driven by trial and error. Time and resources are often invested into ventures that do not bear fruit due to the toxicity of the active pharmaceutical ingredient under investigation. A tool allowing the accurate prediction of toxicity would promote a more rational approach to drug discovery.
## 4.	Datasets <br>
The Toxin and Toxin Target Database (T3DB) contains detailed information (58 columns) about 3,678 toxins such as their chemical formula, weight, appearance, solubility, lethal dose, etc.(https://www.kaggle.com/datasets/ahmedeltom/toxins-and-toxin-target-database)
## 5.	References <br>
1. Directive 2010/63/EU of the European Parliament and of the Council of 22 September 2010 on the Protection of Animals Used for Scientific Purposes Text with EEA Relevance; 2010; Vol. 276. http://data.europa.eu/eli/dir/2010/63/oj/eng (accessed 2023-03-20).

2. Hurst, H. E.; Martin, M. D. 40 - Toxicology. In Pharmacology and Therapeutics for Dentistry (Seventh Edition); Dowd, F. J., Johnson, B. S., Mariotti, A. J., Eds.; Mosby, 2017; pp 603–620. https://doi.org/10.1016/B978-0-323-39307-2.00040-0.

3. Chen, J.; Si, Y.-W.; Un, C.-W.; Siu, S. W. I. Chemical Toxicity Prediction Based on Semi-Supervised Learning and Graph Convolutional Neural Network. Journal of Cheminformatics 2021, 13 (1), 93. https://doi.org/10.1186/s13321-021-00570-8.

4. Zhang, J.; Norinder, U.; Svensson, F. Deep Learning-Based Conformal Prediction of Toxicity. J. Chem. Inf. Model. 2021, 61 (6), 2648–2657. https://doi.org/10.1021/acs.jcim.1c00208.

5. Cremer, J.; Sandonas, L. M.; Tkatchenko, A.; Clevert, D.-A.; Fabritiis, G. de. Equivariant Graph Neural Networks for Toxicity Prediction. ChemRxiv February 8, 2023. https://doi.org/10.26434/chemrxiv-2023-9kb55.

6. Jablonka, K. M.; Schwaller, P.; Ortega-Guerrero, A.; Smit, B. Is GPT-3 All You Need for Low-Data Discovery in Chemistry? ChemRxiv February 14, 2023. https://doi.org/10.26434/chemrxiv-2023-fw8n4.

