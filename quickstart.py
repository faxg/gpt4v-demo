from openai import AzureOpenAI
import os
import json
import time

# OpenAI library
# see: https://platform.openai.com/docs/api-reference/introduction


# AZURE OPENAI
# Docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions
# Swagger: https://github.com/Azure/azure-rest-api-specs/tree/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2023-12-01-preview



api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key= os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSON") or '2023-12-01-preview' # this might change in the future


# Get image_url from user, with a default value
default_image_url = "https://wallpaperaccess.com/full/1260521.jpg"
print(f"Enter image URL (Press Enter to use default: {default_image_url}):")
user_input = input().strip()
# If user_input is not empty, use it as image_url. Otherwise, use the default value.
image_url = user_input if user_input else default_image_url


client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

# Response object see: https://platform.openai.com/docs/api-reference/chat/object
start_time = time.time()
response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": "Describe this picture:" 
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ] } 
    ],
    max_tokens=2000 
)

print(response.choices[0].message.content)

end_time = time.time()

response_time = end_time - start_time
print(f"\n\nResponse time: {response_time} seconds")