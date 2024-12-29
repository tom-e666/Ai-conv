import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import tensorflow as tf
from keras.src.layers import TextVectorization
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model and vectorizer
model = tf.keras.models.load_model('spam_model.keras')
df = pd.read_csv("spam.csv", encoding='latin-1')
X = np.array(df['v2'])

# Initialize the vectorizer
max_features = 10000  # Vocabulary size
sequence_length = 200  # Maximum length of input sequences
vectorizer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
vectorizer.adapt(X)  # Fit the vectorizer on the entire dataset

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]
# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows only these origins to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type)
)

class Message(BaseModel):
    msg: str
@app.post("/predict")
async def predict(message: Message):
    # Preprocess the input message
    text_seq = [message.msg]
    text_vectorized = vectorizer(text_seq)

    # Predict the class (0 = ham, 1 = spam)
    prediction = model.predict(text_vectorized)
    flag = 1 if prediction[0][0] > 0.5 else 0  # You can adjust the threshold
    return {"msg": message.msg, "flag": flag}  # Return the flag as part of the response
