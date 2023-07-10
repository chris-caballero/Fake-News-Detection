import regex as re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def preprocessing_fn(x):
    x = x.lower()
    x = re.sub(r'[^\w\s]', ' ', x)
    x = re.sub(r'\b\w*\d\w*\b', ' ', x)
    x = remove_stopwords(x)
    x = re.sub(' +', ' ', x)
    
    return x