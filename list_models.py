from google import genai
import os

# Client automatically picks up GOOGLE_API_KEY from the environment
client = genai.Client()

print("Available models:")
print("-" * 40)

for model in client.models.list():
    print(model.name)
