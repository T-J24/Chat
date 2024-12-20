import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from text_generation import Client

# Load environment variables
load_dotenv(find_dotenv())  # Read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Hugging Face client setup
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)

# Function to generate response from the AI
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

# Gradio interface
demo = gr.Interface(
    fn=generate, 
    inputs=[gr.Textbox(label="Prompt"), gr.Slider(label="Max new tokens", value=20, maximum=1024, minimum=1)], 
    outputs=[gr.Textbox(label="Completion")]
)

# Launch the interface
gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT1']))
