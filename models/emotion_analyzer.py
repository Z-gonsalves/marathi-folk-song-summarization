# Emotion keywords derived from the Marathi Folk Songs dataset

EMOTIONS = {

    "Heroism": [
        "शिवाजी", "वीर", "शहाजी", "तानाजी", "किल्ला",
        "सरदार", "तलवार", "स्वराज्य", "मावळे", "युद्ध"
    ],

    "Patriotism": [
        "महाराज", "देश", "राजा", "मराठा",
        "काँग्रेस", "ध्वज", "भूमी", "मातृभूमी"
    ],

    "Devotion": [
        "विठ्ठल", "हरि", "राम", "देव",
        "देवा", "नाम", "तुका", "मुक्ताई",
        "पांडुरंग", "कृष्ण", "भक्त"
    ],

    "Love": [
        "जीव", "चांदणी", "पोरी", "बाई",
        "माझ्या", "तुला", "रूप", "मदन"
    ],

    "Joy": [
        "आनंद", "गाऊ", "खेळ", "नाच",
        "उत्सव", "सुख", "गजर"
    ],

    "Sadness": [
        "दुःख", "रड", "विरह", "एकटा",
        "गेली", "नको", "अश्रू"
    ],

    "Anger": [
        "क्रोध", "राग", "शत्रू", "लढ",
        "मोड", "वैर", "युद्ध"
    ]
}

from collections import Counter

def detect_emotions(text):

    words = text.split()

    scores = Counter()

    for emotion, keywords in EMOTIONS.items():

        for word in words:
            if word in keywords:
                scores[emotion] += 1

    if len(scores) == 0:
        return "Neutral", "-", 0

    total = sum(scores.values())

    ranked = scores.most_common()

    primary = ranked[0][0]

    if len(ranked) > 1:
        secondary = ranked[1][0]
    else:
        secondary = "-"

    confidence = round((ranked[0][1] / total) * 100)

    return primary, secondary, confidence