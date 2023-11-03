#img2text
# import Secrets
from transformers import pipeline
import streamlit as st
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import openai, OpenAI

import requests

# openai_api_key = Secret.OPENAI_API_KEY
# Hf_api_key = Secret.HuggingFace_api_key

openai_api_key = st.secrets["OPENAI_API_KEY"]
Hf_api_key = st.secrets["HuggingFace_api_key"]

os.environ['OPENAI_API_KEY'] = openai_api_key
# function to convert image to text
def img2text(url):
    image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
    
    text = image_to_text(url)
    
    print(f"The image says {text}")
    
    return text[0]["generated_text"]
    
    # captioner("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png")
## [{'generated_text': 'two birds are standing next to each other '}]

#llm
# function to convert Text to story
def generate_story(scenario, mood):
    template = """
    you are a good story teller:
    You can generate a good story based ona simple narrative, the story should be around 30 words and
    should be funny and happy
    
    CONTEXT : {scenario} 
    MOOD : {mood}
    STORY:
    """
    prompt = PromptTemplate(template = template, input_variables = ["scenario","mood"])
    
    llm = OpenAI(openai_api_key= openai_api_key, temperature = 0.9)
   
    story_llm = LLMChain(llm = llm,  prompt = prompt)
    
    story = story_llm.predict(scenario = scenario, mood = mood)
            
    return story
    
#text to speech
# function to convert story to audio
def text2speech(message):
  

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {Hf_api_key}"}
    print(f"Bearer {Hf_api_key}")

    payload = {
         "inputs": message,
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    with open('audio.flac','wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="Your image to story")
    st.header("Turn image into story")
    
    upload_file = st.file_uploader("choose an image..", type=["png", "jpg","jpeg"])
    
    if upload_file is not None:
        bytes_data = upload_file.getvalue()
        print(upload_file.name)
        full_path = os.path.join(os.path.dirname(__file__), 'pics', upload_file.name)
        print(f"The full path is {full_path}")
        with open(full_path, "wb") as file:
            file.write(bytes_data)
        st.image(upload_file, caption='uploaded image', use_column_width= True)
        
        mood = st.selectbox('Choose the mood', ('Happy','Sad','Scary', 'silly', 'Spooky'))     
        # scenario = img2text(upload_file.name)
        scenario = img2text(full_path)
        
        updated_scenario = f"{scenario} - {mood}"
        print(f"Updated scenario is {updated_scenario}")
        
        story = generate_story(scenario, mood)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
            
        with st.expander("story"):
            st.write(story)
            
        st.audio("audio.flac")
        print("Done")
    
    
    
if __name__ == '__main__':
    main()