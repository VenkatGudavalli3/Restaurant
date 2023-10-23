# Restaurant Review Sentiment Analysis

This GitHub repository contains the code for a Machine Learning project that focuses on sentiment analysis of restaurant reviews using Natural Language Processing (NLP) techniques. The project aims to predict the sentiment (positive or negative) of restaurant reviews by analyzing the text of the reviews.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Setup and Dependencies](#setup-and-dependencies)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Introduction

Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone behind a piece of text. In this project, we apply sentiment analysis to restaurant reviews to automatically classify them as positive or negative based on the text content.

## Project Overview

1. **Data Import**: The project begins by importing the dataset from a TSV file (`Restaurant_Reviews.tsv`) using pandas. This dataset contains restaurant reviews and their corresponding labels (0 for negative and 1 for positive).

2. **Text Preprocessing**: The text data is preprocessed to clean and prepare it for analysis. This involves removing non-alphabetical characters, converting text to lowercase, and applying stemming to reduce words to their root form. Stop words are also removed from the text.

3. **Feature Extraction**: We create a Bag of Words (BoW) representation of the text data using the `CountVectorizer` from scikit-learn. This step converts the text into numerical features that can be used for machine learning.

4. **Model Training**: We use a Naive Bayes classifier (`GaussianNB`) to train a machine learning model on the preprocessed and feature-extracted data.

5. **Model Evaluation**: The model's performance is evaluated using a confusion matrix and accuracy score to assess its ability to classify restaurant reviews accurately.

## Setup and Dependencies

To run this project, you will need Python and the following libraries:

- pandas
- re
- nltk
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas re nltk scikit-learn
```

## Data

The dataset used in this project is stored in the file `Restaurant_Reviews.tsv`. It contains 1000 restaurant reviews along with their corresponding labels (0 for negative, 1 for positive).

## Preprocessing

Text preprocessing is a crucial step in NLP. In this project, we clean the text data by removing non-alphabetical characters, converting text to lowercase, and applying stemming. Stop words are also removed from the text to reduce noise in the data.

## Feature Engineering

To create numerical features from text data, we use the Bag of Words (BoW) representation. This representation converts text into a matrix of word frequencies, allowing us to train machine learning models on the data.

## Model Training

We use a Gaussian Naive Bayes classifier to train a model for sentiment analysis. Naive Bayes is a probabilistic algorithm commonly used in text classification tasks.

## Evaluation

The model's performance is evaluated using a confusion matrix and accuracy score. The confusion matrix provides insight into the model's ability to correctly classify reviews as positive or negative, while the accuracy score quantifies its overall accuracy.

Feel free to explore the code and adapt it for your own sentiment analysis projects. If you have any questions or suggestions, please don't hesitate to reach out.
