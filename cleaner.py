import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import argparse
import sys

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


def main():
    parser = argparse.ArgumentParser(description="Clean CSV text column by removing stopwords and lemmatizing.")
    parser.add_argument("input", nargs="?", default="data/ai-vs-human-text/balanced_ai_human_prompts.csv",
                        help="Input CSV file (relative to script or absolute).")
    parser.add_argument("-o", "--output", default="data/ai-vs-human-text-clean-testing.csv",
                        help="Output CSV file (relative to script or absolute).")
    args = parser.parse_args()

    # Resolve paths relative to the script directory when given as relative paths
    script_dir = Path(__file__).parent
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = (script_dir / in_path).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (script_dir / out_path).resolve()

    if not in_path.exists():
        print(f"ERROR: Input CSV not found at: {in_path}")
        sys.exit(1)

    try:
        data = pd.read_csv(in_path)
    except Exception as e:
        print(f"Failed to read CSV ({in_path}): {e}")
        sys.exit(1)

    data['generated'] = data['generated'].astype(int)

    # modify the text by removing stop words and lemmatizing
    data['text'] = data['text'].apply(preprocess)

    try:
        data.to_csv(out_path, index=False)
        print("Cleaned data saved to", out_path)
    except Exception as e:
        print(f"Failed to write output CSV ({out_path}): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
