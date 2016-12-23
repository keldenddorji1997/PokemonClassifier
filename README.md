# PokemonClassifier
Predicts the type of Pokemon from its stats <br/>
This is a submission for Siraj's Pokemon Classifier Contest : https://www.youtube.com/watch?v=0xVqLJe9_CY <br/>
The dataset is taken from Kaggle : https://www.kaggle.com/abcsds/pokemon

# Dependencies

Theano (pip install theano) <br/>
NumPy (pip install numpy) <br/>
Pandas (pip install pandas) <br/>

# Demo

Run in terminal: <br/>
$ python PokemonClassifier.py

# Choosing the right features

Before programming the neural networks, to select only the contributing features and to separate the overfitting ones, first I excluded the Total column as it is anyways evaluated using the other features and hence, would overfit. The rest of the dataset (HP, Attack, Defense, Spl. Atk, Spl. Def, Generation, Speed columns) were fed to an Extra Trees Classifier using sklearn which have been commented out in PokemonClassifier.py <br/>
<br/>
It gave the feature importances as: <br/>
HP : 0.15094212 <br/> 
Attack : 0.15045182 <br/> 
Defense : 0.14142085 <br/> 
Spl Atk : 0.16435534 <br/> 
Spl Def : 0.14503864 <br/> 
Speed : 0.15237957 <br/>
Generation : 0.09541167 <br/> 
<br/>
As the feature importance of Generation is relatively low, it should be dropped as a feature which also makes sense as the pokemon's type doesn't actually matter from which pokemon generation series it has come from.  

# Results

The neural network built on Theano is trained to an accuracy of 32.917% in about 3 minutes on CPU. After training, the user can input the stats of a pokemon, and the model will predict its type. <br/>

I thank Siraj Raval for the contest and video explanations, and Alberto Barradas for the Pokemon dataset. 

