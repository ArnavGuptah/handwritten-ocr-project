import re
import spacy
from spellchecker import SpellChecker

# Load models once at startup
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()


def fix_spelling(text):
    """
    Fix spelling mistakes word by word.
    Skips numbers, punctuation, and proper nouns.
    """
    words = text.split()
    corrected = []

    misspelled = spell.unknown(words)

    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if clean.lower() in misspelled and clean.isalpha():
            suggestion = spell.correction(clean.lower())
            if suggestion:
                # Preserve original capitalisation
                if clean.isupper():
                    corrected.append(word.replace(clean, suggestion.upper()))
                elif clean[0].isupper():
                    corrected.append(word.replace(clean, suggestion.capitalize()))
                else:
                    corrected.append(word.replace(clean, suggestion))
            else:
                corrected.append(word)
        else:
            corrected.append(word)

    return " ".join(corrected)


def extract_entities(text):
    """
    Extract named entities using spaCy.
    Finds people, dates, times, places, organisations etc.
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": ent.label_,
            "description": spacy.explain(ent.label_),
        })

    return entities


def extract_keywords(text):
    """
    Extract important keywords — nouns and verbs, no stopwords.
    """
    doc = nlp(text)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.pos_ in ("NOUN", "VERB", "PROPN")
        and len(token.text) > 2
    ]
    return list(set(keywords))


def detect_structure(text):
    """
    Detect basic structure in the note:
    - Bullet points
    - Numbered lists
    - Headings (short lines ending with colon)
    - Dates and times
    """
    lines = text.strip().split("\n")
    structure = []

    date_pattern = re.compile(
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"
    )
    time_pattern = re.compile(r"\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b")
    bullet_pattern = re.compile(r"^[\-\*\•]\s+")
    numbered_pattern = re.compile(r"^\d+[\.\)]\s+")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_type = "text"

        if bullet_pattern.match(line):
            line_type = "bullet"
        elif numbered_pattern.match(line):
            line_type = "numbered"
        elif line.endswith(":") and len(line) < 50:
            line_type = "heading"
        elif date_pattern.search(line):
            line_type = "date"
        elif time_pattern.search(line):
            line_type = "time"

        structure.append({
            "line": line,
            "type": line_type,
        })

    return structure


def process(text, correct_spelling=True):
    """
    Full NLP pipeline on raw OCR text.
    Returns cleaned text + all extracted information.
    """
    original = text

    # Fix spelling
    corrected = fix_spelling(text) if correct_spelling else text

    # Extract info
    entities = extract_entities(corrected)
    keywords = extract_keywords(corrected)
    structure = detect_structure(corrected)

    return {
        "original_text": original,
        "corrected_text": corrected,
        "entities": entities,
        "keywords": keywords,
        "structure": structure,
        "word_count": len(corrected.split()),
        "line_count": len([s for s in structure if s["type"] != "heading"]),
    }