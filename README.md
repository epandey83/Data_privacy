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

• The mask_names function accepts a text input and masks any named entities of type "PERSON" in it.

• Specifying the location of the sound file that will be played as an alarm sound.

• Running nltk.download('stopwords') to download stopwords from nltk.

•	The get_input method is used to obtain user input. It requests input from both the doctor and the patient, uses the mask_names function to mask names in the input, then returns the masked inputs.

•	To preprocess the data, use the preprocess_data function. It eliminates stopwords and vectorizes the supplied data using TF-IDF.\\

•	The train_model function, which uses Multinomial Naive Bayes to train the machine learning model.

•	The test_model function is used to run the trained model on new data and calculate the expected probability.

•	The count_word_occurrences function counts the number of times a certain word appears in a collection of sentences.

•	use the replace_sentence function to duplicate the original sentence list and replace a specified sentence with a replacement sentence.
Considerations

•	To tokenize and parse text, the code makes use of the spaCy library. Before running the code, make sure you have the spaCy English model (en_core_web_sm) installed.

•	The Beck Depression Inventory (BDI) questions are used in the code to measure depression. For simplicity, only three questions are now asked, but you may expand it to include more questions for a more complete examination.

•	By using Laplace noise, differential privacy is applied to the BDI score computation. The epsilon value can be changed to get the appropriate amount of anonymity.

