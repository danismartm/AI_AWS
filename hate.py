from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el modelo de análisis de sentimientos
sentiment_model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_checkpoint)

# Texto de ejemplo (poner en inglés)
text = "You make me feel bad"

# Tokenización del texto
inputs = tokenizer(text, return_tensors="pt")

# Obtener las predicciones del modelo
with torch.no_grad():
    outputs = model(**inputs)

# Obtener los logits y aplicar softmax para obtener probabilidades
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Mostrar la probabilidad para cada clase
print(f'Probabilidades [negativo, positivo]: {probabilities.tolist()}')

# Determinar si el texto es positivo o negativo
if torch.argmax(probabilities) == 1:
    print("Sentimiento positivo")
else:
    print("Sentimiento negativo")




