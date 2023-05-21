import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
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

# # define the function for masking names in the input
# def mask_names(text):
#     pattern = r'\b[A-Z][a-z]+\b'  # pattern to match names
#     masked_text = re.sub(pattern, '*****', text)
#     return masked_text

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
#replace sentence creates copy for original sentence and then use it
def replace_sentence(sentences, i):
    replaced_sentences = sentences.copy()
    replacement_sentence = "This is a replacement sentence."
    replaced_sentences[i] = replacement_sentence
    return replaced_sentences

# define the main function for running the application
def main():
    # define the chat sentences and labels for training
    # Define the word to count
    # word = "depressed"
    sentences = [
        "Hi am John,I am not feeling well",
        "I feel suicidal",
        "I am feeling depressed",
        "I am so stressed out",
        "I am good",
        "I am better",
        "I am okay",
        "I can't sleep at night",
        "I have lost interest in activities",
        "I am constantly worried",
        "I have difficulty concentrating",
        "I feel exhausted all the time",
        "I am experiencing panic attacks",
        "I cry frequently",
        "I am having suicidal thoughts",
        "I feel hopeless and helpless",
        "I don't enjoy things I used to",
        "I feel overwhelmed",
        "I have trouble making decisions",
        "I have no energy",
        "I am irritable and easily agitated",
        "I am irritable and easily agitated",
        "I am irritable and easily agitated"
    ]

    labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]  # 1 for mental health decline
    # Calculate sensitivity using L1 norm
    # sensitivity = max(abs(count_word_occurrences(sentences, word)) - abs(count_word_occurrences(replace_sentence(sentences, i), word)) for i in range(len(sentences)))
    sensitivity = max([sentences.count(sentence) for sentence in sentences])
    print(sensitivity)
    epsilon = 0.2
    private_counts = [sentences.count(sentence) + laplace(scale=sensitivity/epsilon) for sentence in sentences]

    # preprocess the training data
    X_train, tfidf = preprocess_data(sentences)
    y_train = np.array(private_counts).round().astype(int)

    print(y_train)

    # train the model
    model = train_model(X_train, y_train)
    # run the interactive application
    print("💻 | Welcome to the mental health assessment app.")
    while True:
        # get input from the user
        doctor_input, patient_input = get_input()
        # combine the input into a single sentence
        sentence = doctor_input + ' ' + patient_input
        # preprocess the sentence
        X_test = tfidf.transform([sentence])
        # test the model on the new sentence
        y_prob = test_model(model, X_test)
        # output the result
        bdi_score = 0
        print(f"💻 | Based on your input, the probability of a decline in mental health is: {y_prob[0]*1000:.2f}%\n")
        # print(f"💻 | Based on your input, the probability of a decline in mental health is: {y_prob[0]}")
        # check if the probability exceeds 70%
        if y_prob[0] > 0.70:
         print("Mental health decline probability exceeds 70%!")
        # you can also add code here to send an alert to the doctor
        # ask specific questions to determine the person's symptoms
        anxiety = False
        depression = False
        print("💻 | To help assess your mental health, please answer the following questions:")
        # ask anxiety questions
        print("💻 | Anxiety questions:")
        if input("❓ | Do you feel anxious or worried? (yes or no): ").lower() == "yes":
            anxiety = True
        if input("❓ | Are you experiencing any physical symptoms such as rapid heart rate, sweating, or trembling? (yes or no): ").lower() == "yes":
            anxiety = True
        # ask depression questions
        print("💻 | Depression questions:")
        if input("❓ | Do you feel sad, empty, or hopeless? (yes or no): ").lower() == "yes":
            depression = True
        if input("❓ | Are you experiencing any changes in appetite or sleep patterns? (yes or no): ").lower() == "yes":
            depression = True
        if input("❓ | Have you lost interest in activities you used to enjoy? (yes or no): ").lower() == "yes":
            depression = True
        if input("❓ | Are you having any thoughts of self-harm or suicide? (yes or no): ").lower() == "yes":
            depression = True
        # calculate the BDI score based on the person's symptoms
        if anxiety:
            print("💻 | Based on your symptoms, you may be experiencing anxiety.")
        if depression:
            print("💻 | Based on your symptoms, you may be experiencing depression.")
            print("Please answer the following questions from the Beck Depression Inventory (BDI):")
            question_num = 1
            while question_num <= 3:  # ask 3 questions for simplicity
                if question_num == 1:
                    print("❓ |  Do you often feel sad or down?")
                elif question_num == 2:
                    print("❓ |  Do you feel guilty or worthless?")
                elif question_num == 3:
                    print("❓ | Have you lost interest in things you used to enjoy?")
                response = input("💻 | Your response (yes or no): ")
                if response.lower() == "yes":
                    bdi_score += 1
                question_num += 1
         # add differential privacy with epsilon = 1
        # epsilon=0.9
        bdi_score1=bdi_score+y_prob
        # print("bdi_score:",bdi_score)
        # bdi_score += np.random.laplace(loc=0, scale=1/epsilon)
            # output the BDI score
        print(f"Your BDI score is: {bdi_score}\n")

        if bdi_score <= 0 and bdi_score < 0:
            print("🟢 | Your BDI score suggests minimal depression.")
        elif bdi_score > 0 and bdi_score <= 1 :
            print("🟢 | Your BDI score suggests mild depression.")
        elif bdi_score > 1 and bdi_score <= 2:
            print("🟡 | Your BDI score suggests moderate depression.")
        else:
            print("🔴 | Your BDI score suggests severe depression. Please seek immediate professional help.")
            print("🚨 | Notification Set")
            #playsound(sound_file_path)
            p = multiprocessing.Process(target=playsound, args=(sound_file_path,))
            p.start()
            stop_alarm = input("Do you want to stop the alarm? Y/N")
            if stop_alarm == "Y":
                p.terminate()
if __name__ == '__main__':
    main()
