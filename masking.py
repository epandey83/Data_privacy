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

# Example usage
answer = input("Enter your answer: ")
masked_answer = mask_names(answer)
print("Masked answer:", masked_answer)
