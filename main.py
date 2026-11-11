from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-small')
model = AutoModel.from_pretrained('facebook/dinov2-with-registers-small')

inputs = processor(images=image, return_tensors="pt")
print(inputs.pixel_values.shape)

# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

