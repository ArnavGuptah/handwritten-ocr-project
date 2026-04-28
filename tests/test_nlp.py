from app.nlp.nlp_processor import process

raw_text = """Meeting Notes - 12/01/2024
Attendees: John Smith, Sarah Connor

Agenda:
- Discuss projct timeline
- Review budjet proposal
- Next meetng at 3:00pm

Action items:
1. John to send repport by Friday
2. Sarah to contect the client
"""

result = process(raw_text)

print("CORRECTED TEXT:")
print(result["corrected_text"])

print("\nENTITIES FOUND:")
for e in result["entities"]:
    print(" ", e["type"], "->", e["text"])

print("\nKEYWORDS:")
print(" ", result["keywords"])

print("\nSTRUCTURE:")
for s in result["structure"]:
    print(" ", s["type"], "->", s["line"])