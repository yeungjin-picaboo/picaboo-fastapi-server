from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv
from stability_sdk import client
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os 
import boto3
import logging
import io
import openai
import psycopg2
import uuid
import warnings
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

load_dotenv(verbose=True)

openai.api_key= os.getenv('OPENAPI_KEY')

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = os.getenv('STABILITY_KEY')

#AWS
def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3
s3 = s3_connection()


#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #모든 Origin 허용
    allow_credentials=True, #인증정보 허용
    allow_methods=["*"], #HTTP Method 허용
    allow_headers=["*"], #HTTP Header허용
)

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-xl-beta-v2-2-2",
    )


class Diary(BaseModel):
    content: str

class requestData(BaseModel):
    prompt: str
    userId: int

@app.get("/")
def root_main():
    return {"message": "Picaboo AI Server"}

@app.get('/api/diaries/emotion/{content}')
def summarize_diary(content: str):
    #response_summary = openai.Completion.create(
    #    model="text-davinci-002",
    #    prompt=f"Summarize the following text: '{content}'.",
    #    max_tokens=50,
    #)
    #summary = response_summary.choices[0].text.strip()

    response_emotion = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Extract emotion from the following text: '{content}'.",
        max_tokens=5,
    )

    emotion = response_emotion.choices[0].text.strip()
    return {'emotion':emotion}
    #return {'summary': summary, 'emotion': emotion}

@app.get("/api/diaries/picture/{content}")
def make_picture(content:str):
    answers = stability_api.generate(
    prompt=content,
    seed=992446758, 
    steps=30,
    cfg_scale=8.0,      
    width=512,
    height=512,
    samples=1, 
    #sampler=generation.SAMPLER_K_DPMPP_2M
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                filename = str(uuid.uuid4()) + ".png"
                filepath = os.path.join('img', filename)
                img.save(filepath)
                try:
                    s3.upload_file(filepath,"picaboodiaryimage",filename)
                except Exception as e:
                    print(e)
    
    return filename

