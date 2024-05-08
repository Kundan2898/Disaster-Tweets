#Kaggle Disaster Tweets Classification Project Overview

This project focuses on classifying tweets from Twitter during disasters. The goal is to build a machine learning model that can accurately classify whether a tweet is about a real disaster or not.

Dataset
The dataset used for this project is the "Real or Not? NLP with Disaster Tweets" dataset from Kaggle. You can find the dataset here. It contains the following files:

train.csv: Contains the training data.
test.csv: Contains the test data.
Requirements
To run the code, you need the following libraries:

pandas
numpy
scikit-learn
nltk

Preprocessing
Before building the classification model, the text data in the tweets is preprocessed using the following steps:

Tokenization: Splitting the text into words.
Removing Stopwords: Removing common words that do not carry much information.
Stemming or Lemmatization: Reducing words to their base or root form.
Vectorization: Converting text data into numerical data using techniques like TF-IDF.

Model
For this project, a Support Vector Machine (SVM) classifier is used. SVM is chosen for its effectiveness in text classification tasks.

Files
disaster_tweets_using_sv.ipynb: Jupyter notebook containing the code for data preprocessing, model building, and evaluation, train dataset, test dataset, submission file.
README.md: This file.

Usage

1. Download the dataset from the provided link.
Clone this repository.

2. Place the dataset files (train.csv and test.csv) in the same directory as the notebook.

3. Open and run the notebook disaster_tweets_using_sv.ipynb in Jupyter Notebook.

4. Follow the instructions in the notebook to preprocess the data, build the model, and make predictions.

Results

The model achieves an accuracy of approximately 80% on the test data.