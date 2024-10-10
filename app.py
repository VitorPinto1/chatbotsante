import pyttsx3
import vosk
import sounddevice as sd
import json
import queue
import time
from flask import Flask, request, jsonify
import noisereduce as nr
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Importer les données d'entraînement et les conversations
from training_data import train_sentences, train_labels

# Initialiser Flask
app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

# Charger le modèle Vosk pour le français
MODEL_PATH = "vosk-model-small-fr-0.22"
model = vosk.Model(MODEL_PATH)

# Configurer la synthèse vocale
engine = pyttsx3.init()

# Préparation des données pour Naive Bayes
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_train, train_labels, test_size=0.2, random_state=42)

# Entraîner le modèle Naive Bayes
model_ml = MultinomialNB()
model_ml.fit(X_train, y_train)

# File d'attente pour le flux audio
q = queue.Queue()

# Tableau des récompenses
reward_table = defaultdict(int)


# --------------------- Fonctions du chatbot ---------------------
def preprocess_text(text):
    stop_words = set(stopwords.words('french'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def detect_intent_ml(user_input):
    try:
        user_input_processed = preprocess_text(user_input)
        X_test = vectorizer.transform([user_input_processed])
        predicted_intent = model_ml.predict(X_test)[0]
        return predicted_intent
    except Exception as e:
        return None

def speak(text):
    print(f"Robert: {text}")
    engine.say(text)
    engine.runAndWait()

def handle_intent(intent, user_input=None, previous_intent=None):
    if intent == "greet":
        return "Bonjour, comment puis-je vous aider aujourd'hui ?", None

    elif intent == "exit":
        return "Au revoir, prenez soin de vous !", "exit"

    elif intent == "ask_injury_fall":
        response = "Si la douleur est intense, consultez un médecin. Sinon, reposez-vous et appliquez de la glace."
        speak("Avez-vous besoin d'appeler SOS ?")
        response = listen()
        next_intent = detect_intent_ml(preprocess_text(response))
        if next_intent == "confirm_sos_call":
            return "D'accord, j'appelle SOS. Veuillez rester calme.", None
        elif next_intent == "decline_sos_call":
            return "D'accord, prenez soin de vous.", None
        else:
            return "Je n'ai pas bien compris, pouvez-vous reformuler ?", None

    elif intent == "ask_flu_symptoms":
        return "Les symptômes de la grippe incluent la fièvre, la toux, et des douleurs musculaires. Avez-vous de la fièvre ?", None
    
    elif intent == "ask_flu_fever":
        if previous_intent == "ask_flu_symptoms":
            return "Pouvez-vous me dire votre température actuelle ?", None
        else:
            return "Avez-vous de la fièvre ?", None

    elif intent == "ask_flu_treatment":
        return "Pour traiter la grippe naturellement, reposez-vous, buvez beaucoup de liquides, et prenez des remèdes naturels comme le miel ou l'infusion de gingembre.", None

    elif intent == "unknown":
        return "Je n'ai pas bien compris, pouvez-vous reformuler ?", None

    else:
        return "Je ne sais pas comment répondre à cela. Pouvez-vous reformuler ?", None

# --------------------- Fonction d'écoute (mode vocal) ---------------------
def callback(indata, frames, time, status):
    audio_data = np.frombuffer(indata, dtype=np.int16)
    filtered_data = nr.reduce_noise(y=audio_data, sr=16000)
    q.put(filtered_data.tobytes())

def listen():
    time.sleep(1)
    recognizer = vosk.KaldiRecognizer(model, 16000)
    
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("Vous pouvez parler maintenant...")
        speech_text = ""
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                speech_text = json.loads(result).get('text', '')
                if speech_text:
                    print(f"Texte reconnu : {speech_text}")
                    return speech_text.lower()
                else:
                    speak("Je n'ai pas bien compris, pouvez-vous répéter ?")


def vocal_chat():
    current_intent = None
    first_time = True

    try:
        if first_time:
            speak("Bonjour, je suis Robert, votre assistant vocal. Comment puis-je vous aider ?")
            first_time = False  # Une seule fois au démarrage
        while True:
            # Écouter et analyser l'intention
            user_input = listen()
            intent = detect_intent_ml(preprocess_text(user_input))

            if intent == "exit":
                speak("Au revoir !")
                break

            # Réponse du chatbot basée sur l'intention détectée
            response, next_intent = handle_intent(intent, user_input)
            
            # Si l'intention suivante est None, l'utilisateur doit répondre
            if next_intent:
                current_intent = next_intent
            else:
                current_intent = None

            speak(response)

    except Exception as e:
        speak(f"Une erreur s'est produite : {str(e)}")


# --------------------- API Flask ---------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    is_vocal = data.get('isVocal', False)

    if not user_input and not is_vocal:
        return jsonify({"response": "Je n'ai pas bien compris, pouvez-vous reformuler ?"})

    # Si mode vocal activé, démarrer la conversation vocale
    if is_vocal:
        vocal_chat()
        return jsonify({"response": "Mode vocal activé."})
    
    # Sinon, détecter l'intention basée sur le texte
    intent = detect_intent_ml(preprocess_text(user_input))
    
    # Gérer l'intention détectée
    response, next_intent = handle_intent(intent, user_input)

    return jsonify({"response": response})



@app.route('/chat-sos', methods=['POST'])
def chat_sos():
    data = request.get_json()
    user_input = data.get('message', '')
    is_vocal = data.get('isVocal', False)

    if not user_input and not is_vocal:
        return jsonify({"response": "Je n'ai pas bien compris, pouvez-vous reformuler ?"})

    # Simuler des actions spécifiques pour le SOS
    if is_vocal:
        vocal_chat()
        return jsonify({"response": "Mode vocal activé pour SOS."})

    # Exemple de réponse spécifique pour le SOS
    response = f"Message SOS reçu : {user_input}. Nous allons prendre les mesures nécessaires."

    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(debug=True)
