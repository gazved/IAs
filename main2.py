from transformers import pipeline
from deep_translator import GoogleTranslator

tradutor = GoogleTranslator(source= 'pt', target= 'en')

classifier = pipeline("text-classification", model="Falconsai/offensive_speech_detection")
texto = "se mata seu arrombado sem mae ."
traducao = tradutor.translate(texto)

result = classifier(traducao)
print(result)
