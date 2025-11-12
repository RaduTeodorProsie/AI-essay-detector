from cleaner import preprocess
import joblib

model = joblib.load("nb_scratch_model.joblib")
print("Enter text to classify:")
raw_text = input()

clean_text = preprocess(raw_text)

# the function used by the trainer
from trainer import predict_one

predicted_class = predict_one(model, clean_text)

ans = "AI" if predicted_class == 1 else "Human"
print("Predicted class: " + ans)
