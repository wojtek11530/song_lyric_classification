def assess_emotion_four_classes(arousal: float, valance: float) -> str:
    if arousal >= 0.5 and valance >= 0.5:
        return 'happy'
    elif arousal >= 0.5 and valance < 0.5:
        return 'angry'
    elif arousal < 0.5 and valance >= 0.5:
        return 'relaxed'
    else:
        return 'sad'


def assess_emotion_two_classes(valance: float) -> str:
    if valance >= 0.5:
        return 'positive'
    else:
        return 'negative'
