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

#### Explanation of code for Psychoeducation feature
K-Anonymity applied on the dataset containing the following attributes :-
Key attributes : UserID, Email and Name
Quasi-Identifiers : Country, Age, Gender and Relationship Status
Sensitive attributes : Chat Sharing Preferences, Documents Shared With Therapist and Symptoms

#### Symptom Tracker
The code shared above is a Python script called symptom_tracker.py that generates synthetic data for symptom tracking. It uses the pandas library for data manipulation and randomization.

Code Overview
The script begins by importing the necessary libraries: pandas, random, and datetime.

It defines the number of users (num_users) and the start and end years for generating random timestamps.

The random_date() function is defined, which generates a random date within the specified range.

The script creates lists of symptoms and notes to choose from.

A loop is used to generate synthetic data for each user. Random values are selected for user ID, symptom, severity, frequency, note, and timestamp. The generated data is stored in a list called data.

The list data is used to create a Pandas DataFrame named df with appropriate column names.

The DataFrame df is saved to a CSV file named symptom_data.csv using the to_csv() function.

Next, the script defines mappings for severity levels and frequency categories.

The severity values in the DataFrame are mapped to corresponding severity descriptions using a lambda function.

The timestamp values in the DataFrame are rounded to the nearest month.

The month values are mapped to the corresponding season of the year using a dictionary mapping.

The season and year values are combined to create a new column called Timestamp in the format "Season Year".

Unnecessary columns (Month, Season, and Timestamp) are dropped from the DataFrame.

The de-identified DataFrame is saved to a new CSV file named deidentified_symptom_data.csv.

Usage
To use the Symptom Tracker code:

Make sure you have Python installed on your system.

Clone or download the repository containing the symptom_tracker.py file.

Open the script in a Python editor or IDE.

Customize the parameters in the script, such as the number of users, symptoms, severity levels, frequency categories, and notes.

Run the script, either through the editor or by executing the following command in the terminal:

```bash
python3 symptom_tracker.py
```
The script will generate synthetic data for symptom tracking and save it to a CSV file named symptom_data.csv.

Additionally, the script will de-identify the data and save it to a separate CSV file named deidentified_symptom_data.csv.