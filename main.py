import joblib
import tensorflow as tf
from keras.src.layers import TextVectorization
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('spam_model.keras')
df = pd.read_csv("spam.csv", encoding='latin-1')
X = np.array(df['v2'])
max_features = 10000  # Vocabulary size (adjust based on your dataset)
sequence_length = 200  # Maximum length of input sequences
vectorizer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
vectorizer.adapt(X)  # Fit the vectorizer on the entire dataset
# text_seq = ["this is another summer"]  # Example text
# text_vectorized = vectorizer(text_seq)

# prediction = model.predict(text_vectorized)
# print("Prediction (0 = ham, 1 = spam):", prediction[0][0])
from fastapi import FastAPI
from pydantic import BaseModel






class Message(BaseModel):
    msg: str
    clave:str

app = FastAPI()
@app.get("/")
def fn():
    return {"state": "available"}

@app.post("/predict/")
async def predict(message: Message):
    text_seq = [message.msg]
    text_vectorized = vectorizer(text_seq)
    pred = model.predict(text_vectorized)
    res = "spam" if pred > 0.7 else "ham"
    return {"label": res}
