import base64
import os
from typing import Optional
from openai import BaseModel
from pydantic import Field
import requests
import dotenv


# Langchain Imports
from langchain.output_parsers import PydanticOutputParser


dotenv.load_dotenv()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# API key
api_key = os.getenv("OPENAI_API_KEY")


class GPTImageAnalysis(BaseModel):
    message: str = Field(
        description="The advice given by the assistant for the blind person"
    )


def describe(image_path: str) -> Optional[GPTImageAnalysis]:

    try:

        parser = PydanticOutputParser(pydantic_object=GPTImageAnalysis)

        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        format_instructions = parser.get_format_instructions()

        describe_prompt = (
            "Describe in simple words and a short sentence what is in the image, less than 8 words. "
            "Format the sentence in such a way as if you were speaking directly to a blind person. "
            "Also, warn about any potential danger or obstacles in the image only if necessary."
        )

        # Add Format Instructions to the prompt
        describe_prompt += f"\n\n{format_instructions}"

        payload = {
            "model": "gpt-4o",
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an AI assistant for a blind person. "
                                "You received an image and you need to describe it and warn about potential danger or obstacles."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": describe_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        response = response.json()["choices"][0]["message"]["content"]

        data = parser.parse(response)

        print(data)

        return data

    except Exception as e:
        print(e)
        return None
