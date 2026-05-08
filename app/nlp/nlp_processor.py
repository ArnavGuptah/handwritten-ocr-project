import re
from collections import Counter


def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9.\s]', '', text)
    text = " ".join(text.split())
    return text


def extract_keywords(text):
    stopwords = {
        "is", "am", "are", "the", "a", "an", "for",
        "my", "this", "that", "and", "in", "on",
        "of", "to", "it"
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    filtered = [
        w for w in words
        if w not in stopwords
    ]

    freq = Counter(filtered)

    return [word for word, _ in freq.most_common(7)]


def generate_summary(text):
    sentences = re.split(r'[.!?]', text)

    cleaned = [
        s.strip() for s in sentences
        if len(s.strip()) > 20
    ]

    summary = cleaned[:3]

    return summary


def process(text):
    text = clean_text(text)

    keywords = extract_keywords(text)

    summary = generate_summary(text)

    return {
        "corrected_text": text,
        "entities": [],
        "keywords": keywords,
        "summary": summary,
        "word_count": len(text.split())
    }
