from nltk import download, word_tokenize, pos_tag
from nltk.corpus import stopwords, gutenberg, wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter

# Convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(treebank_tag[0], wordnet.NOUN)

# Download required data if not present
for package in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'gutenberg']:
    download(package, quiet=True)

# Read Moby Dick from Gutenberg corpus
text = gutenberg.raw('melville-moby_dick.txt')

# Tokenize and remove stopwords in one go
stop_words = set(stopwords.words("english"))
tokens = [
    word for word in word_tokenize(text.lower())
    if word.isalnum() and word not in stop_words
]

# POS tagging
pos_tags = pos_tag(tokens)

# Count POS frequencies and find the top 5
pos_freq = Counter(tag for word, tag in pos_tags).most_common(5)
print("Most common POS tags and frequencies:", pos_freq)

# Lemmatize using proper POS tags
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [
    (lemmatizer.lemmatize(word, get_wordnet_pos(tag)), tag)
    for word, tag in pos_tags
][:20]
print("\nTop 20 lemmatized tokens:", lemmatized_tokens)

# Plot POS frequencies
plt.figure(figsize=(12, 6))
plt.bar(*zip(*Counter(tag for word, tag in pos_tags).items()))
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of POS')
plt.show()
