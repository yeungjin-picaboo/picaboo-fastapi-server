from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv
from stability_sdk import client
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os 
import io
import openai
import warnings
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

app = FastAPI()

load_dotenv(verbose=True)

openai.api_key= os.getenv('OPENAPI_KEY')

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = os.getenv('STABILITY_KEY')


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

@app.get("/")
def root_main():
    return {"message": "Picaboo AI Server"}

@app.get('/api/diaries/emotion')
def summarize_diary(diary: Diary):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"summary: {diary.content}",
        max_tokens=50,
    )
    summary = response.choices[0].text.strip()
    return {'summary': summary}

@app.get("/api/diaries/picture/{prompt}")
def make_picture(prompt:str):
    answers = stability_api.generate(
    prompt=prompt,
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
                img.save(str(artifact.seed)+ ".png") 
                filename = str(artifact.seed) + ".png"

            #imageUrl = f'http://picaboonftimage.s3.ap-northeast-2.amazonaws.com/{filename}'
    return FileResponse(filename)
    # return imageUrl

