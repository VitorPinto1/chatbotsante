import pyttsx3
import vosk
import sounddevice as sd
import json
import queue
import time
import noisereduce as nr
from conversations import *
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Chemin vers le modèle Vosk pour la langue française
MODEL_PATH = "vosk-model-small-fr-0.22"

# Initialiser le modèle Vosk
model = vosk.Model(MODEL_PATH)

# File d'attente pour le flux audio
q = queue.Queue()

# Configurer la synthèse vocale
engine = pyttsx3.init()

# Fonction pour parler
def speak(text):
    print(f"Robert: {text}")
    engine.say(text)
    engine.runAndWait()

# Fonction callback pour capturer l'audio
def callback(indata, frames, time, status):
    # Convertir indata en tableau NumPy pour le traitement
    audio_data = np.frombuffer(indata, dtype=np.int16)
    
    # Appliquer une réduction de bruit sur l'audio capturé
    filtered_data = nr.reduce_noise(y=audio_data, sr=16000)
    
    # Remettre les données filtrées dans la file d'attente
    q.put(filtered_data.tobytes())

# Fonction pour écouter avec Vosk
def listen():
    time.sleep(1)
    recognizer = vosk.KaldiRecognizer(model, 16000)  # Initialiser le recognizer avec Vosk
    
    # Démarrer l'enregistrement audio avec sounddevice
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("Vous pouvez parler maintenant...")
        speech_text = ""
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                speech_text = json.loads(result).get('text', '')
                if speech_text:  # Si du texte est reconnu
                    print(f"Texte reconnu : {speech_text}")
                    return speech_text.lower()  # Retourne le texte reconnu
        

# Fonction pour gérer les réponses plus flexibles, avec détection de mots-clés
def detect_intent(user_input):
    # Listes de mots-clés pour les réponses positives et négatives
    positive_responses = ["oui", "ouais", "bien sûr", "absolument", "je pense que oui", "tout à fait", "évidemment", "c'est exact", "exactement", "bien entendu"]
    negative_responses = ["non", "pas du tout", "je ne pense pas", "je ne crois pas", "je pense que non", "certainement pas", "absolument pas", "négatif", "sûrement pas"]
    sortir_responses = ["au revoir", "à bientôt", "bye", "ciao", "fin de la conversation", "je dois partir", "stop", "quitter", "terminer", "arrêter"]


    # Vérifier si l'une des réponses positives est dans la réponse utilisateur
    for word in positive_responses:
        if word in user_input:
            return "oui"
    
    # Vérifier si l'une des réponses négatives est dans la réponse utilisateur
    for word in negative_responses:
        if word in user_input:
            return "non"
        
    for word in sortir_responses:
        if word in user_input:
            return "au revoir"
    
    
    # Si aucune intention claire n'est détectée, retourner None
    return None

def handle_open_responses(user_input, flow):
    # Vérifier si des mots-clés spécifiques comme "fièvre", "toux", etc. sont dans la réponse
    for keyword in flow:
        if keyword in user_input:
            return flow[keyword]  # Retourner la réponse spécifique au mot-clé détecté
    return None  # Si aucune correspondance trouvée, retourner None

# Fonction pour poser la question finale
def ask_another_question():
    speak("Avez-vous une autre question ?")
    user_input = listen()

    intent = detect_intent(user_input)
    if intent == "oui":
        vocal_chat()  # Relancer la conversation si l'utilisateur a une autre question
    elif intent == "non":
        speak("Merci pour vos questions. Au revoir !")
    else:
        # Reposer une seule fois si la réponse n'est pas claire
        speak("Je ne suis pas sûr de comprendre. Avez-vous une autre question ?")
        user_input = listen()
        intent = detect_intent(user_input)
        if intent == "oui":
            vocal_chat()
        elif intent == "non":
            speak("Merci pour vos questions. Au revoir !")

    
def handle_flow(flow):
    while True:
        # Poser la question principale
        speak(flow['question'])
        user_input = listen()

        # Utiliser detect_intent pour détecter l'intention
        intent = detect_intent(user_input)

        if intent == "au revoir":
            speak("Au revoir !")
            exit()


        if intent == "oui" and "oui" in flow:
            next_step = flow["oui"]
        elif intent == "non" and "non" in flow:
            next_step = flow["non"]
        else:
            next_step = handle_open_responses(user_input, flow)

        if next_step:
            if isinstance(next_step, dict):  # Si c'est un sous-flow, continuer
                handle_flow(next_step)
            else:
                speak(next_step)  # Si c'est une réponse finale
                ask_another_question()
                break  # Arrêter la boucle après avoir donné la réponse finale
        else:
            speak("Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?")
            continue  
        
      

# Fonction principale de chat vocal avec détection des mots-clés
def vocal_chat():
    global first_time 
    first_time = True
    if first_time:
        # Afficher et dire le message de bienvenue une seule fois
        speak("Bonjour, je suis Robert, votre assistant vocal. Comment puis-je vous aider ?.")
        first_time = False  #
    
    else:
        # Si ce n'est pas la première fois, un autre message est joué
        speak("Comment puis-je vous aider à nouveau ?.")
    
    
    while True:
        user_input = listen()

        intent = detect_intent(user_input)

        if intent == "au revoir":
            speak("Au revoir !")
            exit()

        # Vérification des mots-clés pour lancer un flow spécifique
        if "grippe" in user_input:
            handle_flow(chatbot_flow["symptômes de la grippe"])
        elif "blessure" in user_input:
            handle_flow(chatbot_flow["blessure"])
        elif "au revoir" in user_input:
            speak("Au revoir !")
            exit()
        else:
            speak("Je ne suis pas sûr de comprendre. Pouvez-vous répéter ?")

# Démarrer le chatbot vocal
vocal_chat()

