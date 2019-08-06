# New Bias Classifier
Determine if a news article is biased! This code implements a classifier to take the text of a news article, and guess whether it is left- or right-biased.

![Image of the classifier](screenshot.PNG | width=250)

# Quickstart
main.py implements a flask app to classify the bias in a news article.

# How it Works
For several years, ratings of the bias of common news sources have been accumnulating (we use ratings from www.mediabiasfactcheck.com). If we assume that an article from a given news source, on average, has the same bias as that source in general, then we can access a large corpus of news articles (we use https://www.kaggle.com/snapcrack/all-the-news) and train a machine learning model to take the article as input, and output the average bias of that news sources. Critically, because the classifier is actually using only the text of the article to make this judgment (and is blind to its source), the trained model should be able to classify the bias of articles from many different news sources, whether or not they are included in the mediabias ratings dataset.

This is a sensitive topic, so it's important to be realistic about the performance of this model. Based on experiments with the training corpus, the model has an F1-score of 0.73, which very roughly means that in a five-category guess between "left bias", "left-center bias", "least biased", "right-center biased", and "right biased", the model is right about 3/4 of the time. Thus, it's fair to say that the model is reasonably accurate, but, can still make mistakes. It is also possible that the model could make more mistakes on types of text that the model was not trained on, such as blogs or social media posts. 

# Some Details
The function train_model() should be executed first. This trained an L2-logistic regression classifier to use the text of news articles (represented as tf-idf unigram embeddings) to classify the bias of the article, on a scale from 1 (left bias) --> 5 (right bias). 

The function index() can be thought of as the main pipeline. This function reads the text of an article from an HTML form, classifies its bias using the trained model, and outputs a webpage with explanation of the bias classification.

# Related Projects

A related project is at http://www.areyoufakenews.com/

# Extensions

This classifier can be extended in several ways. Some suggestions:

* More diverse training corpus. The training corpus is quite large but predominantly from several large newspaper sources. Including many more sources (blogs, etc.) can likely improve the model.

* Deep learning. The classifier currently represents words in a way that is blind to their order of occurence. A deep learning approach, especially RNN-based, can likely perform better.

* Article-level labels. The assumption in training the model is that every article from a news source has the same bias. This assumption is reasonable in the abstract but not true for every article. Better training can likely be achieved if news bias can be labeled at the article level.