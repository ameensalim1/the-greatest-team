import sqlite3
import pandas as pd

# Load the CSV into a DataFrame
csv_file = '../data/spotify_data_normalized.csv'  # Replace with your file path
df = pd.read_csv(csv_file)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('songs.db')
cursor = conn.cursor()

# Create the songs table
cursor.execute('''
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY,
    artist_name TEXT NOT NULL,
    track_name TEXT NOT NULL,
    track_id TEXT NOT NULL UNIQUE,
    popularity INTEGER,
    year INTEGER,
    genre TEXT,
    danceability REAL,
    energy REAL,
    key INTEGER,
    loudness REAL,
    mode INTEGER,
    speechiness REAL,
    acousticness REAL,
    instrumentalness REAL,
    liveness REAL,
    valence REAL,
    tempo REAL,
    duration_ms INTEGER,
    time_signature INTEGER
)
''')

rock_songs = cursor.execute(
'''SELECT DISTINCT genre
FROM songs;'''
)
print(rock_songs.fetchall())
# Insert data into the database
df.to_sql('songs', conn, if_exists='replace', index=False)
print("Database created and data imported successfully!")

print("Top 10 songs by popularity:")
# Commit and close
conn.commit()
conn.close()

