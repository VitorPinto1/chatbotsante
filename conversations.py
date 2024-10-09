


reflections = {
    "je suis": "vous êtes",
    "je": "vous",
    "mon": "votre",
    "ma": "votre",
}

# Dictionnaires des réponses simples pour le chatbot
chatbot_responses = {
    "symptômes de la grippe": "Les symptômes de la grippe incluent la fièvre, la toux, et des douleurs musculaires.",
    "température grippe": "La température lors d'une grippe peut dépasser 38°C. Consultez un médecin si elle dépasse 39°C.",
    "traitement grippe": "Le traitement de la grippe inclut le repos, l'hydratation, et parfois des médicaments pour soulager la douleur.",
    "consultation médecin grippe": "Si les symptômes persistent ou s'aggravent, consultez un médecin.",
}

# Dictionnaire imbriqué pour un chatbot plus contextuel
chatbot_flow = {
    "symptômes de la grippe": {
        "question": "Avez-vous des symptômes comme la fièvre, la toux ou des douleurs musculaires ?",
        "oui": {
            "question": "Cela ressemble à des symptômes de la grippe. Votre température est-elle supérieure à 38°C ?",
            "oui": {
                "response": "Votre température est élevée. Assurez-vous de vous reposer, de vous hydrater et de consulter un médecin si elle dépasse 39°C.",
                "question": "Prenez-vous des médicaments pour la fièvre ou les douleurs musculaires ?",
                "oui": "Continuez à suivre les recommandations du médecin. Si vous ne vous sentez pas mieux dans les prochains jours, contactez un professionnel de santé.",
                "non": "Je vous recommande de prendre des médicaments en vente libre pour soulager la fièvre et les douleurs musculaires. Si les symptômes persistent, consultez un médecin."
            },
        
            "non": {
                "response": "Même sans forte fièvre, reposez-vous et hydratez-vous bien. Consultez un médecin si les symptômes persistent.",
            }
        },
        "non": {
            "question": "Si vous ne présentez pas de symptômes de grippe, cela pourrait être un simple rhume. Cependant, surveillez vos symptômes et consultez un médecin en cas de doute."
        }
    },
    "blessure": {
        "question": "Avez-vous une blessure suite à une chute ou une coupure ?",
        "coupure": {
            "question": "Votre coupure est-elle petite ou profonde ?",
            "petite": "Nettoyez la coupure avec de l'eau et du savon, puis appliquez un pansement. Si elle commence à s'infecter, consultez un médecin.",
            "profonde": "Si la coupure est profonde ou saigne abondamment, appliquez une pression et allez immédiatement voir un professionnel de santé."
        },
        "chute": {
            "question": "Ressentez-vous une douleur intense après la chute ?",
            "oui": "Vous pourriez avoir une fracture ou une entorse. Immobilisez la zone et rendez-vous à l'hôpital pour des examens.",
            "non": "Si la douleur est supportable, reposez-vous et appliquez de la glace pour réduire l'enflure. Consultez un médecin si la douleur persiste."
        }
    }
}





training_data = [
    ("bonjour", "greet"),
    ("salut", "greet"),
    ("hello", "greet"),
    ("quels sont les symptômes de la grippe ?", "ask_flu_symptoms"),
    ("comment traiter la grippe ?", "ask_flu_treatment"),
    ("j'ai une blessure", "ask_injury"),
    ("je me suis coupé", "ask_injury"),
    ("au revoir", "goodbye"),
]