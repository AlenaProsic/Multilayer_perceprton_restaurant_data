
# Multi layer perceptron (MLP)
An MLP classifier for making reliable predictions on how expencive a restaurant is expected to be from its menu text.





## Author

- [@AlenaProsic](https://github.com/AlenaProsic)


## Summary
The goal of this project is to provide a  MLP classifier for making reliable predictions on how expencive a restaurant is expected to be from its menu text.
The underlying motivation for building such a classifier can be drawn from the notion of food as a core element of culture and the way and language used in talking about culture as an important tool in the context of cultural research, and thus, food.
## Documentation

[The code can be accessed in a stable version via this link (https://github.com/AlenaProsic/Multilayered_perceprton_restaurant_data)](https://github.com/AlenaProsic/Multilayered_perceprton_restaurant_data)


## Project organization

The project is organized in following modules: 
1. Data module. Where data for train and test is stored. 
2. Evaluation module. Which  contains preparation class for evaluation to store TP, FP, and FN for a given restaurant. 
3. Feature Extraction module. Which contains a class a class to extract various features from restaurants data. 
4. Machine Learning. A module to predict a restaurant's price tag with a Linear Regressor.
5. Preprocessing. Which represents a class to preprocess data for the baseline implementation.
6. Resources. Contains a class to represent a corpus of restaurants.

## Data Description
The dataset is sourced from University of Stuttgart. 
It is comprised of menus from different restaurants. 
There are 5 categories presented in the dataset: 
- Price tag;
- Restaurant name;
- Type of cuisine;
- Location area;
- Dish descriptions;


## Features

A module featureExtraction holding a class featureExtractor to extract features from a restaurant object. 
Features we extract: 
- Bag-of-Words;
- Word length;
- Dish length;
- Menu length;
- Stop words;
- Bigrams.


## Objectives

The main objectives for feature extraction are the following:
- Bag-of-Words containing counts of the tokens present in the menu texts for the restaurants in a given corpus restaurant;
- The feature to extract word length  to word complexity in order to boost $$$$ && $$$ prediction; It counts the number of characters if each token in the restaurant.tokens and ads it as a key feature in the restaurant.features. The underlying concept of the longer and the more complicated word is the more likely the restaurant is going to be expencive;
- The feature extracts dish length from restaurant.menu_text/tokens and computes the average. The underlying  assumption is that the longer the dish is the more likely the restaurant is going to be cheap (the concept of plenty)
- The feature extracts menu length, using restaurants.menu_text. The underlying assumption is that the longer menu is the more likely to be cheap (the concept of plenty and abundance of consumer choice)
- The function extracts bag-of-words features coupled with the list of predefined stop words adopted from nlkt, using restaurant.features BOW for most often occurring words. Stop word list includes function words, like e.g. prepositions, conjunctions ("and"), articles, axilary verbs,  etc. The underlying concept is that function words wont receive more weight due to their high frequency;
- The function extracts bigrams from restaurant.tokens using nltk, creating a list of tuples stored as an ordered dictionary. The underlying concept is that the frequently occuring bigrams might help to boost the performance of a certain class / classes.
## Baseline implementation 

The implementation of the basline includes a class perceptron to represent the architecture of a binary perceptron classifier. Each instance has an attribute weights to store weight vectors as well as the methods get weight and compute score to compute a score for a feature set of a given restaurant. The methods increase feat vector and decrease feat vector provide a way to increase or decrease the feature vector of a given feature set and add a feature to a given dictionary weights, if necessary.
The class MultiClassPerceptron provides a representation of a set of binary classifiers, one for each price tag in the range of $ to $$$$. The method train is used to train the classifier on a training corpus (menu train.txt) with an even numver of epochs. For each restaurant in a randomly shuffled list of restaurants, the methods redict and update are invoked to get a price tag prediction for a restaurant. If necessary, the feature vectors of the perceptrons for a predicted and gold price tag are de- or increased. After each training round, the predictions made are evaluated with our evaluation tool from the package evaluation to see how performance changes. The trained multi-class perceptron is then saved to disk and applied to a development data set(menu dev.set). We report overall micro and macro F-Score as well as per price tag for the performance of each feature.
## Reflections and Future work 

The performance of the multi-class classifier has proved to be slightly heightened, with an overffiting tendency. Given that the forth and third price categories were most underrepresented, the score showed lower results. Future work to improve classifier performance would involve the exploration of possible features
such as word length, language (Language Detection Tool needed), adjectives (PoS-Tagger needed), average length of dish, average length of menu, positive sentiment, the concept of plenty, traditional vs. provenance f food, tf-idf, compounds, words consisting of only one or two letters, bigrams, etc. Another area which can be explored would be Linear Regression in order to investigate the correlation between various variables based on counts of words and the price tag as a dependent variable as well as some control variables like e.g. the cuisine or location.
