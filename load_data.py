import json
import pandas as pd

# Load JSON data
with open('art_artists.json') as f:
    data = json.load(f)

# Extract relevant information
records = []
for artist in data['artists']:
    for artwork in artist['artworks']:
        records.append({
            'artist_name': artist['artist_name'],
            'country': artwork['country'],
            'city': artwork['city']
        })

# Create DataFrame
df = pd.DataFrame(records)

# Save DataFrame to CSV
df.to_csv('artworks.csv', index=False)
