from cleaner import preprocess
import joblib

try:
    model = joblib.load("nb_scratch_model.joblib")
except Exception as e:
    print(f"Could not load model: {e}. You need to use the trainer first")
    exit(0)

print("Enter text to classify:")
raw_text = input()

clean_text = preprocess(raw_text)

# the function used by the trainer
from trainer import predict_one

predicted_class = predict_one(model, clean_text)

ans = "AI" if predicted_class == 1 else "Human"
print("Predicted class: " + ans)
