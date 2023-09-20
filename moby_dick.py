# Import NLTK, requests and matplotlib libraries
import nltk
import requests
import matplotlib

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read the Moby Dick file from the Gutenberg dataset
moby_dick_url = "https://www.gutenberg.org/files/2701/2701-0.txt"
moby_dick_text = requests.get(moby_dick_url).text

# Tokenize the entire book
tokens = nltk.word_tokenize(moby_dick_text)
# Filter out the stopwords
stopwords = nltk.corpus.stopwords.words('english')
filtered_tokens = [token for token in tokens if token not in stopwords]

# Tag the different parts of speech for each word
tagged_tokens = nltk.pos_tag(filtered_tokens)

# Count and display the 5 most common parts of speech and their total counts
pos_counts = nltk.FreqDist(tag for (word, tag) in tagged_tokens)
print(pos_counts.most_common(5))

# Lemmatize the top 20 tokens
lemmatizer = nltk.stem.WordNetLemmatizer()
top_tokens = [word for (word, count) in pos_counts.most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in top_tokens]
print(lemmatized_tokens)

# Plot a bar chart to visualize the frequency of POS
nltk.FreqDist.plot(pos_counts, title="Frequency of POS in Moby Dick", cumulative=False)
