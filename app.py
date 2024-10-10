"""
import pyttsx3
import vosk
import sounddevice as sd
import json
import queue
import time
from flask import Flask, request, jsonify, render_template
import noisereduce as nr
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import defaultdict
from flask_cors import CORS

# Importer les données d'entraînement et les conversations
from training_data import train_sentences, train_labels

# Initialiser Flask
app = Flask(__name__)
CORS(app)

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


@app.route('/')
def index():
    return render_template('index.html')  # Rendre le fichier HTML


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


def text_chat():
    current_intent = None
    first_time = True

    try:
        if first_time:
            print("Robert: Bonjour, je suis Robert, votre assistant. Comment puis-je vous aider ?")
            first_time = False  
        else:
            print("Robert: Comment puis-je vous aider à nouveau ?")
        
        while True:
            # Si aucune intention n'est en cours, demander une entrée utilisateur
            if current_intent is None:
                user_input = input("Vous: ")  # Remplace la capture vocale par une entrée texte
                if not user_input.strip():
                    print("Robert: Aucune réponse détectée. En attente de votre entrée...")
                    continue  # Réécouter l'utilisateur s'il n'y a pas de réponse
            else:
                user_input = current_intent
            
            # Utiliser le modèle machine learning pour détecter l'intention
            intent = detect_intent_ml(preprocess_text(user_input))

            # Vérifier si l'intention est une sortie
            if intent == "exit":
                print("Robert: Au revoir !")
                break
            
            # Gérer l'intention détectée
            response, next_intent = handle_intent(intent, user_input)

            # Si l'intention est inconnue ou si la réponse est un échec, demander à reformuler
            if intent == "unknown" or "Je ne sais pas comment répondre" in response:
                print(f"Robert: {response}")
                continue  # Réécouter sans poser une autre question

            # Si la réponse est valide, la donner
            print(f"Robert: {response}")

            # Si aucune intention de suivi n'est présente, réécouter l'utilisateur
            if next_intent is None:
                # Laisser le chatbot simplement répondre sans poser de question supplémentaire
                user_input = input("Vous: ")  # Demander une autre entrée utilisateur
                if not user_input.strip():
                    print("Robert: Aucune réponse détectée. En attente de votre entrée...")
                    continue  # Réécouter l'utilisateur
                current_intent = detect_intent_ml(preprocess_text(user_input))  # Détecter la prochaine intention utilisateur
                continue
            else:
                current_intent = next_intent  # Continuer avec la prochaine intention

    except Exception as e:
        print(f"Robert: Une erreur s'est produite : {str(e)}")


# Démarrer le chatbot basé sur texte
text_chat()

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

    return jsonify({"response": "Response from /chat"})



@app.route('/chat-sos', methods=['POST'])
def chat_sos():
    data = request.get_json()  # Parse the incoming JSON data
    if not data:
        return jsonify({"error": "Invalid input"}), 400
    
    user_input = data.get('message', '')
    is_vocal = data.get('isVocal', False)

    if not user_input and not is_vocal:
        return jsonify({"response": "Je n'ai pas bien compris, pouvez-vous reformuler ?"}), 400

    # Simulate a response for SOS
    response = f"Message SOS reçu : {user_input}. Nous allons prendre les mesures nécessaires."

    return jsonify({"response": response}), 200  



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Importer les données d'entraînement et les conversations
from training_data import train_sentences, train_labels

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes CORS pour toutes les origines

# Vectoriser les phrases
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_sentences)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=0.2, random_state=42)

# Entraîner le modèle Naive Bayes
model_ml = MultinomialNB()
model_ml.fit(X_train, y_train)

# Table des récompenses pour chaque intention
reward_table = defaultdict(int)  # Les intentions commenceront à 0 point

def preprocess_text(text):
    stop_words = set(stopwords.words('french'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def detect_intent_ml(user_input):
    try:
        user_input_processed = preprocess_text(user_input)
        X_input = vectorizer.transform([user_input_processed])
        predicted_intent = model_ml.predict(X_input)[0]
        return predicted_intent
    except Exception as e:
        print(f"Une erreur s'est produite lors de la détection d'intention : {str(e)}")
        return None

def handle_intent(intent, user_input=None, previous_intent=None):
    if intent == "greet":
        return "Bonjour, comment puis-je vous aider aujourd'hui ?", None

    elif intent == "exit":
        return "Au revoir, prenez soin de vous !", "exit"

    elif intent == "ask_injury_fall":
        response = "Si la douleur est intense, consultez un médecin. Sinon, reposez-vous et appliquez de la glace."
        # Ensuite, poser la question SOS
        next_question = "Avez-vous besoin d'appeler SOS ?"
        return f"{response} {next_question}", "awaiting_sos_response"

    elif intent == "confirm_sos_call":
        return "D'accord, j'appelle SOS. Veuillez rester calme.", None

    elif intent == "decline_sos_call":
        return "D'accord, prenez soin de vous.", None

    elif intent == "ask_flu_symptoms":
        return "Les symptômes de la grippe incluent la fièvre, la toux et des douleurs musculaires. Avez-vous de la fièvre ?", None

    elif intent == "ask_flu_fever":
        if previous_intent == "ask_flu_symptoms":
            return "Pouvez-vous me dire votre température actuelle ?", None
        else:
            return "Avez-vous de la fièvre ?", None

    elif intent == "ask_flu_treatment":
        return "Pour traiter la grippe naturellement, reposez-vous, buvez beaucoup de liquides et prenez des remèdes naturels comme le miel ou l'infusion de gingembre.", None

    elif intent == "unknown":
        return "Je n'ai pas bien compris, pouvez-vous reformuler ?", None

    else:
        # Pour toute autre intention non prévue
        return "Je ne sais pas comment répondre à cela. Pouvez-vous reformuler ?", None

@app.route('/')
def index():
    return render_template('index.html')  # Vous pouvez créer un fichier index.html pour le front-end

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"response": "Aucune donnée reçue."}), 400

    user_input = data.get('message', '')
    if not user_input:
        return jsonify({"response": "Veuillez fournir un message."}), 400

    previous_intent = data.get('previous_intent', None)

    # Détection de l'intention
    intent = detect_intent_ml(user_input)
    if not intent:
        intent = "unknown"

    # Gestion de l'intention
    response, next_intent = handle_intent(intent, user_input, previous_intent)

    # Mise à jour des récompenses (exemple simplifié)
    update_rewards(intent, success=True)

    return jsonify({
        "response": response,
        "intent": intent,
        "next_intent": next_intent
    }), 200

def update_rewards(intent, success):
    if success:
        reward_table[intent] += 1  # Récompense positive
    else:
        reward_table[intent] -= 1  # Récompense négative

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
