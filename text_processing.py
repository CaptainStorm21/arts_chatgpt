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

    # Convert the field to lowercase for case insensitivity
    field_lower = field.lower()

    # Initialize a list to store results
    results = []

    # Iterate through artists and artworks
    for artist in data['artists']:
        # Convert field values to lowercase for comparison
        artist_field_value = artist.get(field_lower, '').lower()
        
        # Check if artist_field_value matches the value or if value is empty (match any)
        if artist_field_value == value.lower() or value == '':
            # Perform tokenization (NLTK)
            text = artist['biography']
            tokens = word_tokenize(text)
            result = {
                "artist_name": artist['artist_name'],
                "tokenized_text": tokens,
                "complete_sentence": ' '.join(tokens)
            }
            
            # Perform text processing (SpaCy)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            pos_tags = [(token.text, token.pos_) for token in doc]
            result["pos_tags"] = pos_tags

            results.append(result)

    return results if results else None  # Re

# Example usage to look up any artist by 'artist_name'
if __name__ == "__main__":
    results = lookup_and_process('artist_name', '')
    if results:
        for result in results:
            print("Artist Name:", result["artist_name"])
            print("Tokenized Text:", result["tokenized_text"])
            print("Complete Sentence:", result["complete_sentence"])
            print("POS Tags:")
            for token, pos in result["pos_tags"]:
                print(f"{token} - {pos}")
            print("--------------")
    else:
        print("No artists found.")