import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix
import nltk
import unicodedata
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import random
import seaborn as sns
import contractions
# Load Dataset
import kagglehub
path=kagglehub.dataset_download("kazanova/sentiment140")
print(path)
import os
path_s='/root/.cache/kagglehub/datasets/kazanova/sentiment140/versions/'
files=os.listdir(path)
print(files)
file_path='/root/.cache/kagglehub/datasets/kazanova/sentiment140/versions/2/training.1600000.processed.noemoticon.csv'
df_w=pd.read_csv(file_path,encoding='ISO-8859-1',header=None)
df_w.columns=['target', 'ids', 'date', 'flag', 'user', 'text']
df_w.head()


# Sample 40% of the Data
df = df_w.sample(frac=0.4, random_state=42)  # Randomly select 40% of rows
print("Subset Size:", len(df))
print("\nSample Data:\n", df.head())

# Check Sentiment Distribution
sentiment_counts = df['target'].value_counts()
print("\nSentiment Label Distribution:")
print(sentiment_counts)

# Plot Sentiment Distribution
sentiment_counts.plot(kind='bar', title='Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
from textblob import TextBlob
# Function to classify sentiment using TextBlob
def classify_sentiment_textblob(tweet):
    # Analyze sentiment polarity
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply the TextBlob sentiment classification
df['textblob_sentiment'] = df['text'].apply(classify_sentiment_textblob)

# Display a sample of the original text and their classified sentiment
print("\nTextBlob Sentiment Classification (Sample):")
print(df[['text', 'textblob_sentiment']].head(10))

# Display the distribution of TextBlob sentiment results
textblob_sentiment_counts = df['textblob_sentiment'].value_counts()
print("\nTextBlob Sentiment Distribution:")
print(textblob_sentiment_counts)

# Plot the distribution of sentiment results
textblob_sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], title='TextBlob Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
import random
from textblob import TextBlob

# Function to analyze polarity and subjectivity
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Randomly sample 20 tweets for analysis
random_tweets = df.sample(n=20, random_state=42)

# Apply the function to the sampled tweets
sentiment_results = random_tweets['text'].apply(analyze_sentiment)

# Create a DataFrame to display sentiment results
results_df = pd.DataFrame({
    'Tweet': random_tweets['text'],
    'Polarity': sentiment_results.apply(lambda x: x[0]),
    'Subjectivity': sentiment_results.apply(lambda x: x[1])
})

# Reset index for neat display
results_df = results_df.reset_index(drop=True)

# Display the results
print("\nRandom Tweets with Polarity and Subjectivity Analysis:")
print(results_df)

# Plot polarity and subjectivity for the random tweets
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Polarity'], results_df['Subjectivity'], color='purple', alpha=0.7)
plt.title('Polarity vs. Subjectivity of Random Tweets')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.grid(True)
plt.show()
import re
import string

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Expand contractions
    contractions = {"don't": "do not", "i'm": "i am", "it's": "it is"}
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    # Remove URLs, @mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    return text

# Apply Preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)
print("\nPreprocessed Text Example:\n", df['clean_text'].head())
import re
import string

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Expand contractions
    text = fix(text)
    # Remove URLs, @mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    return text

# Apply Preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)
print("\nPreprocessed Text Example:\n", df['clean_text'].head())

# Tokenize Words
df['tokens'] = df['clean_text'].apply(word_tokenize)
print("\nTokenized Words Example:\n", df['tokens'].head())
from nltk.tokenize import sent_tokenize

# Sentence Tokenization
df['sentences'] = df['clean_text'].apply(sent_tokenize)
print("\nTokenized Sentences Example:\n", df['sentences'].head())
from nltk.tokenize import sent_tokenize

# Sentence Tokenization
df['sentences'] = df['clean_text'].apply(sent_tokenize)

# Display full tokenized sentences for a few tweets
print("\nTokenized Sentences Example:")
for i in range(5):  # Adjust the range to display more or fewer examples
    print(f"Tweet {i+1}:")
    for sentence in df['sentences'].iloc[i]:
        print(f"  - {sentence}")
    print()  # Add a blank line for better readability
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

# Transform Text
bow_matrix = vectorizer.fit_transform(df['clean_text'])

# Display BoW Results
print("\nVocabulary (BoW):", vectorizer.get_feature_names_out())
print("\nBoW Matrix Shape:", bow_matrix.shape)
# Reduce the dataset to a smaller random subset for demonstration
df_sample = df.sample(n=10, random_state=42)  # Only 10 tweets for inspection

# Initialize CountVectorizer with limited vocabulary size
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=20)  # Top 20 words

# Transform the sample dataset
bow_matrix = vectorizer.fit_transform(df_sample['clean_text'])

# Convert BoW matrix to a DataFrame for easy viewing
vocabulary = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vocabulary, index=df_sample.index)

# Display the BoW matrix
print("\nBoW Matrix (Small Sample):\n")
print(bow_df)

# Optional: Display the sample tweets alongside their BoW representation
print("\nSample Tweets and BoW Representation:")
for idx, row in bow_df.iterrows():
    print(f"Tweet {idx}: {df_sample['clean_text'][idx]}")
    print(f"BoW: {row.to_dict()}")
    print("-" * 40)
# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

# Transform Text
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

# Display TF-IDF Results
print("\nVocabulary (TF-IDF):", tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Reduce the dataset to a smaller random subset for demonstration
df_sample = df.sample(n=10, random_state=42)  # Only 10 tweets for inspection

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=20)  # Limit to top 20 words

# Transform the sample dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sample['clean_text'])

# Convert TF-IDF matrix to a DataFrame for easy viewing
vocabulary = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocabulary, index=df_sample.index)

# Display the TF-IDF matrix
print("\nTF-IDF Matrix (Small Sample):\n")
print(tfidf_df)

# Optional: Display the sample tweets alongside their TF-IDF representation
print("\nSample Tweets and TF-IDF Representation:")
for idx, row in tfidf_df.iterrows():
    print(f"Tweet {idx}: {df_sample['clean_text'][idx]}")
    print(f"TF-IDF: {row.to_dict()}")
    print("-" * 40)

# Tokenized Text for Word2Vec
tokenized_text = df['tokens'].tolist()

# Train Word2Vec Model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Check Vocabulary Size
print("\nWord2Vec Vocabulary Size:", len(word2vec_model.wv.index_to_key))

# Get Vector for a Word
word = "love"
print(f"Vector for '{word}':\n", word2vec_model.wv[word])

# Find Similar Words
similar_words = word2vec_model.wv.most_similar('love')
print("\nWords Similar to 'love':", similar_words)

# Load GloVe Embeddings - Updated to handle file not found error
glove_path = 'glove.6B.100d.txt'  
embeddings_index = {}

try:
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    # Example: Get Vector for a Word
    word = "hate"
    print(f"Vector for '{word}' (GloVe):\n", embeddings_index.get(word))

except FileNotFoundError:
    print(f"Error: The file '{glove_path}' was not found. Please ensure the file is downloaded and placed in the correct directory.")

from collections import Counter
# Before Preprocessing
raw_text = ' '.join(df['text'])
raw_words = raw_text.split()
unique_raw_words = set(raw_words)
word_counts_raw = Counter(raw_words)

# After Preprocessing
processed_text = ' '.join(df['clean_text'])
processed_words = processed_text.split()
unique_processed_words = set(processed_words)
word_counts_processed = Counter(processed_words)

# Display the number of unique words
print(f"Number of Unique Words Before Preprocessing: {len(unique_raw_words)}")
print(f"Number of Unique Words After Preprocessing: {len(unique_processed_words)}")

# Visualize Word Frequency Distribution (Before and After Preprocessing)
plt.figure(figsize=(12, 6))

# Top 20 most common words before preprocessing
raw_top_20 = word_counts_raw.most_common(20)
raw_words, raw_freqs = zip(*raw_top_20)
plt.bar(raw_words, raw_freqs, alpha=0.7, label='Before Preprocessing')

# Top 20 most common words after preprocessing
processed_top_20 = word_counts_processed.most_common(40)
processed_words, processed_freqs = zip(*processed_top_20)
plt.bar(processed_words, processed_freqs, alpha=0.7, label='After Preprocessing')

plt.xticks(rotation=45)
plt.title("Word Frequency Distribution Before and After Preprocessing")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Select a smaller subset of the data for demonstration
df_sample = df.sample(n=200, random_state=42)

# Initialize Vectorizers
bow_vectorizer_raw = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
bow_vectorizer_processed = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_vectorizer_raw = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_vectorizer_processed = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

# Fit and Transform (Before Preprocessing)
bow_raw = bow_vectorizer_raw.fit_transform(df_sample['text'])
tfidf_raw = tfidf_vectorizer_raw.fit_transform(df_sample['text'])

# Fit and Transform (After Preprocessing)
bow_processed = bow_vectorizer_processed.fit_transform(df_sample['clean_text'])
tfidf_processed = tfidf_vectorizer_processed.fit_transform(df_sample['clean_text'])

# Convert to DataFrames for Comparison
bow_raw_df = pd.DataFrame(bow_raw.toarray(), columns=bow_vectorizer_raw.get_feature_names_out())
bow_processed_df = pd.DataFrame(bow_processed.toarray(), columns=bow_vectorizer_processed.get_feature_names_out())

tfidf_raw_df = pd.DataFrame(tfidf_raw.toarray(), columns=tfidf_vectorizer_raw.get_feature_names_out())
tfidf_processed_df = pd.DataFrame(tfidf_processed.toarray(), columns=tfidf_vectorizer_processed.get_feature_names_out())

# Display Example Word Scores (Choose a Few Words for Demonstration)
example_words = ['happy', 'love', 'great', 'me']

# Create Comparison Table
comparison = {
    "Word": example_words,
    "BoW Raw": [bow_raw_df[word].sum() if word in bow_raw_df.columns else 0 for word in example_words],
    "BoW Processed": [bow_processed_df[word].sum() if word in bow_processed_df.columns else 0 for word in example_words],
    "TF-IDF Raw": [tfidf_raw_df[word].sum() if word in tfidf_raw_df.columns else 0 for word in example_words],
    "TF-IDF Processed": [tfidf_processed_df[word].sum() if word in tfidf_processed_df.columns else 0 for word in example_words],
}

comparison_df = pd.DataFrame(comparison)

# Display Results
print("\nComparison of Word Scores Before and After Preprocessing:")
print(comparison_df)

# Plot the Changes in Weights
comparison_df.set_index("Word").plot(kind="bar", figsize=(12, 6))
plt.title("Word Scores in BoW and TF-IDF (Before and After Preprocessing)")
plt.ylabel("Score")
plt.show()
# Flatten the cleaned text into a single list of words
all_words = ' '.join(df['clean_text']).split()

# Count word occurrences
word_counts = Counter(all_words)

# Check specific words
print("Occurrences in the dataset:")
print("happy:", word_counts.get('happy', 0))
print("me:", word_counts.get('me', 0))
print("love:", word_counts.get('love', 0))
