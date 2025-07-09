import streamlit as st
import numpy as np
import pickle
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Load resources ---
nltk.download('stopwords')
nltk.download('punkt')

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("multitask_nlp_model.h5")

# Set max sequence length (same as training)
max_length = 50

# Stopwords
stop_words = set(stopwords.words('english'))

# Preprocess function
def remove_stopwords(text):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])

# Label mappings
emotion_labels_text = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels_text = ['sexual_violence', 'physical_violence', 'emotional_violence', 'Harmful_traditional_practice', 'economic violence']
hate_labels_text = ['offensive speech', 'Neither', 'Hate Speech']
major_labels = ['Emotion', 'Violence', 'Hate']

# Inference function
def classify_text(input_text):
    cleaned = remove_stopwords(input_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')

    pred = model.predict({
        'emotion_input': padded,
        'violence_input': padded,
        'hate_input': padded
    })

    emotion_pred = np.argmax(pred[0], axis=1)[0]
    violence_pred = np.argmax(pred[1], axis=1)[0]
    hate_pred = np.argmax(pred[2], axis=1)[0]

    # Major label based on highest score
    major_idx = np.argmax([np.max(pred[0]), np.max(pred[1]), np.max(pred[2])])
    major_label = major_labels[major_idx]

    if major_label == 'Emotion':
        sub_label = emotion_labels_text[emotion_pred]
    elif major_label == 'Violence':
        sub_label = violence_labels_text[violence_pred]
    else:
        sub_label = hate_labels_text[hate_pred]

    return major_label, sub_label

# --- Streamlit UI ---
st.title("🧠 Detect Emotions, Violence, Hate from Text")

user_input = st.text_area("Enter your text:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        major, sub = classify_text(user_input)
        st.success(f"**Major Label:** {major}")
        st.info(f"**Sub Label:** {sub}")
