from PIL import Image
from transformers import pipeline

# Carrega a imagem
img = Image.open("test01.png")

# Configura o pipeline para classificação de imagens
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# Classifica a imagem
result = classifier(img)

# Exibe o resultado
print(result)
