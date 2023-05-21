import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from playsound import playsound
import math 
import multiprocessing
from numpy.random import laplace
import re
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def mask_names(text):
    # Tokenize the input text
    doc = nlp(text)
    
    # Iterate over each token in the document
    masked_text = []
    for token in doc:
        if token.ent_type_ == "PERSON":
            # If the token is a named entity of type PERSON, mask it
            masked_text.append("[MASKED]")
        else:
            # Otherwise, keep the original token
            masked_text.append(token.text)
    
    # Join the tokens back into a single string
    masked_text = " ".join(masked_text)
    return masked_text

# specify the path of the sound file
sound_file_path = "alarm.mp3"

# download stopwords for preprocessing
nltk.download('stopwords')

# define the function for getting input from the user
def get_input():
    # get input from the doctor
    doctor_input = input("💻 | How can I help you today?")
    # get input from the patient
    print("\n")
    patient_input = input("🧍| ")
    # mask names in the input
    masked_doctor_input = mask_names(doctor_input)
    masked_patient_input = mask_names(patient_input)

    print(masked_patient_input)
    return masked_doctor_input, masked_patient_input

# define the function for preprocessing the data
def preprocess_data(data):
    stop_words = stopwords.words('english')
    tfidf = TfidfVectorizer(stop_words=stop_words)
    X = tfidf.fit_transform(data)
    return X, tfidf

# define the function for training the model
def train_model(X, y):
    clf = MultinomialNB()
    clf.fit(X, y)
    return clf

# define the function for testing the model on new data
def test_model(model, X_test):
    y_prob = model.predict_proba(X_test)
    return y_prob[:, 0]

# Define the function to count word occurrences
def count_word_occurrences(sentences, word):
    count = 0
    for sentence in sentences:
        count += sentence.lower().count(word.lower())
    return count

# replace sentence creates a copy of the original sentence and then uses it
def replace_sentence(sentences, i):
    replaced_sentences = sentences.copy()
    replacement_sentence = "This is a replacement sentence."
    replaced_sentences[i] = replacement_sentence
    return replaced_sentences

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report

# define the main function for running the application
def main():
    # define the chat sentences and labels for training
    sentences = [
        "Hi am John,I am not feeling well",
        "I feel suicidal",
        "I feel depressed",
        "I feel anxious",
        "I feel happy and content",
        "I feel great"
    ]
    labels = [
        1, 1, 1, 1, 0, 0
    ]

    # preprocess the training data
    X_train, tfidf = preprocess_data(sentences)
    
    # Define the number of epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        
        # train the model
        model = train_model(X_train, labels)

        # generate test data
        test_sentences = [
            "I feel sad",
            "I feel stressed",
            "I feel motivated"
        ]
        test_labels = [
            1, 1, 0
        ]
        X_test = tfidf.transform(test_sentences)
        
        # test the model
        y_pred_prob = test_model(model, X_test)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)

        # evaluate the model
        accuracy, precision, recall, f1, report = evaluate_model(model, X_test, test_labels)
        print("Evaluation Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print("\nClassification Report:")
        print(report)

        # play sound if the prediction indicates a decline in mental health
        for i, pred in enumerate(y_pred):
            if pred == 1:
                playsound(sound_file_path)

        print()

if __name__ == "__main__":
    main()
