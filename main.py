from fastapi import FastAPI
from fastapi.responses import FileResponse
import os 
import io
import warnings
from PIL import Image
from dotenv import load_dotenv
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

load_dotenv(verbose=True)

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = os.getenv('STABILITY_KEY')

app = FastAPI()

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-xl-beta-v2-2-2",
    )

@app.get("/")
def root_main():
    return {"message": "Picaboo AI Server"}

@app.get("/api/diaries/picture/{prompt}")
def make_picture(prompt:str):
    answers = stability_api.generate(
    prompt="expansive landscape rolling greens with blue daisies and weeping willow trees under a blue alien sky, artstation, masterful, ghibli",
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

    #return {"prompt":prompt, "filename":filename}
    return FileResponse(filename)