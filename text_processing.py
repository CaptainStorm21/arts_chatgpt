import json
import nltk
from nltk.tokenize import WordPunctTokenizer
import spacy

# Download NLTK resources if needed
nltk.download('punkt')

# Function to perform lookup and process text
def lookup_and_process(query):
    # Load JSON data
    with open('art_artists.json') as f:
        data = json.load(f)

    # Initialize a list to store results
    results = []

    # Convert query to lowercase for case insensitivity
    query_lower = query.lower()

    # Iterate through artists and artworks
    for artist in data['artists']:
        # Check if query matches artist's name or any artwork's title
        artist_name_lower = artist['artist_name'].lower()
        artworks = artist['artworks']

        # Perform tokenization (WordPunctTokenizer for better handling of punctuation)
        text = artist['biography']
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(text)

        # Check if the query matches the artist's name or any artwork's title
        if (query_lower in artist_name_lower or any(query_lower in artwork['title'].lower() for artwork in artworks)):
            result = {
                "artist_name": artist['artist_name'],
                "tokenized_text": tokens,
                "complete_sentence": ' '.join(tokens),
                "artworks": [artwork for artwork in artworks if query_lower in artwork['title'].lower() or query_lower in artist_name_lower]
            }

            # Perform text processing (SpaCy)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            pos_tags = [(token.text, token.pos_) for token in doc]
            result["pos_tags"] = pos_tags

            results.append(result)

    return results if results else None  # Return None if no matches found

# Function to handle user interaction
def main():
    while True:
        # Prompt user to enter artist's name or artwork title
        query = input("Enter artist's name or artwork title (or press Enter to search all): ").strip()

        # Lookup and process based on user input
        results = lookup_and_process(query)

        if results:
            for result in results:
                print("Artist Name:", result["artist_name"])
                print("Brief Biography:", result["complete_sentence"])
                print("Artwork:")
                for artwork in result["artworks"]:
                    print(f"Title: {artwork['title']}")
                    print(f"Description: {artwork['description']}")
                    print(f"Year Created: {artwork['year_created']}")
                    print(f"Location: {artwork['city']}, {artwork['country']}")
                    print('----------------------')
                print()  # Add a vertical space between each result

            # Prompt user to find another artist or artwork title
            choice = input("Would you like to find another artist or artwork title? (Yes/No): ").strip().lower()
            if choice != 'yes':
                print("Exiting program.")
                break
        else:
            print("No artists or artworks found.")
            break

if __name__ == "__main__":
    main()
