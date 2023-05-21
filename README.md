# Data_privacy
This repository contains source code for a mental health assessment application that analyzes text input and provides an evaluation of the user's mental health using natural language processing methods and machine learning. The program includes features for preprocessing data, building a machine learning model, and assessing mental health issues based on user replies.

Dependencies
•	numpy (as np)
•	pandas (as pd)
•	nltk
•	sklearn.feature_extraction.text (TfidfVectorizer)
•	sklearn.naive_bayes (MultinomialNB)
•	sklearn.metrics (classification_report)
•	playsound
•	math
•	multiprocessing
•	numpy.random (laplace)
•	re
•	spacy
please install these dependencies before running the code.

Usage
•	Load the spaCy English model using nlp = spacy.load("en_core_web_sm").
•	mask_names function that takes a text input and masks any named entities of type "PERSON" in the text.
•	Specifying the path of the sound file to be played for an alarm sound.
•	Downloading stopwords from nltk by running nltk.download('stopwords').
•	get_input function to get input from the user. It prompts for input from both the doctor and the patient, masks names in the input using the mask_names function, and returns the masked inputs.
•	preprocess_data function to preprocess the data. It removes stopwords and performs TF-IDF vectorization on the input data.
•	train_model function to train the machine learning model using Multinomial Naive Bayes.
•	test_model function to test the trained model on new data and obtain the predicted probabilities.
•	count_word_occurrences function to count the occurrences of a specific word in a list of sentences.
•	replace_sentence function to create a copy of the original sentences list and replace a specific sentence with a replacement sentence.
Considerations
•	The code utilizes the spaCy library to tokenize and process text. Make sure to have the spaCy English model installed (en_core_web_sm) before running the code.
•	The code uses the Beck Depression Inventory (BDI) questions for assessing depression. Currently, only three questions are asked for simplicity, but you can extend it to include more questions for a more comprehensive assessment.
•	Differential privacy is applied to the BDI score calculation by adding Laplace noise. The epsilon value can be adjusted for the desired level of privacy.

