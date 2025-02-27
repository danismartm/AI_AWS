from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Usar un modelo público para detectar texto tóxico
toxicity_model_checkpoint = "unitary/toxic-bert"

# Cargar el tokenizador y el modelo
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_checkpoint)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_checkpoint)

# Frase a analizar
text = "Eres una persona terrible y te odio."

# Tokenizar la entrada
toxicity_input_ids = toxicity_tokenizer(text, return_tensors="pt").input_ids

# Obtener los logits (probabilidades sin normalizar)
logits = toxicity_model(toxicity_input_ids).logits
print(f'logits[not hate, hate]: {logits.tolist()[0]}')

# Calcular las probabilidades
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [not hate, hate]: {probabilities}')

# Obtener los datos de "not hate" (la recompensa)
not_hate_index = 0  # Asumiendo que "not hate" es el primer valor en los logits
nothate_reward = logits[:, not_hate_index].tolist()
print(f'reward (value of "not hate" logit): {nothate_reward}')

