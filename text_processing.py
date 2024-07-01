import json
import nltk
from nltk.tokenize import word_tokenize
import spacy

# Download NLTK resources if needed
nltk.download('punkt')

# Function to perform lookup and process text
def lookup_and_process(field, value):
    # Load JSON data
    with open('art_artists.json') as f:
        data = json.load(f)

    # Iterate through artists and artworks
    for artist in data['artists']:
        if field in artist and artist[field] == value:
            # Perform tokenization (NLTK)
            text = artist['biography']
            tokens = word_tokenize(text)
            print("Tokenized Text:", tokens)
            
            # Concatenate tokens into a sentence
            sentence = ' '.join(tokens)
            print("Complete Sentence:", sentence)

            # Perform text processing (SpaCy)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for token in doc:
                print(token.text, token.pos_)

            # Break after first match (or adjust logic for multiple matches)
            break

# Example usage
if __name__ == "__main__":
    lookup_and_process('name', 'Leonardo da Vinci')
