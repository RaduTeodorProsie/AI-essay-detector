import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


data = pd.read_csv("data/ai-vs-human-text/AI_Human.csv")


data['generated'] = data['generated'].astype(int)

#modify the text by removing stop words and lemmatizing
data['text'] = data['text'].apply(preprocess)


clean_path = "data/ai-vs-human-text-clean.csv"
data.to_csv(clean_path, index=False)

print("Cleaned data saved to", clean_path)
