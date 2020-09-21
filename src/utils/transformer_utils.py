from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


def clean_text(text, max_len=128):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    words = [word for word in word_tokens if word.isalpha()]

    filtered_words = [w for w in words if not w in stop_words]

    return ' '.join(filtered_words)[0:max_len-1].lower()
