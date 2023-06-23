from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv
from stability_sdk import client
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
import os 
import boto3
import logging
import io
import openai
import nltk
nltk.download('punkt')
nltk.download('emotion')
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

#nltk.download('vader_lexicon')
#sentimentNltk = SentimentIntensityAnalyzer()

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

# 이미지 저장할 폴더 경로
img_folder = 'img'
# 유지할 이미지 최대 갯수
max_image_count = 20

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
    response_summary = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text: '{content}'.",
        max_tokens=50,
    )
    summary = response_summary.choices[0].text.strip()
    print("summary:" + summary)

    #response_emotion = openai.Completion.create(
    #    model="text-davinci-002",
    #    prompt=f"Extract emotion from the following text: '{content}'.",
    #    max_tokens=5,
    #)

    #emotion = response_emotion.choices[0].text.strip()
    #return {'emotion':emotion}
    #return {'summary': summary, 'emotion': emotion}
    # VADER 사전 감정 추출
    # sentiments = sentimentNltk.polarity_scores(content)
    #emotion = None
    #if sentiments['pos'] > sentiments['neg']:
    #    emotion = "happy"
    #elif sentiments['neg'] > sentiments['pos']:
    #    emotion = "sad"
    #else:
    #    emotion = "neutral"
    #print(sentiments["pos"])
    #return emotion
    # 감정 카테고리에 대한 단어 정의
    happy_words = ['happy','great', 'joyful', 'excited','nice','today is happy', 'very happy', "so i'm very happy"]
    good_words = ['good', 'wonderful','today is good','study good']
    neutral_words = ['neutral', 'not bad', ]
    bad_words = ['bad', 'negative', 'unhappy', 'disappointed','bad today']
    confused_words = ['confused']
    angry_words = ['angry', 'very angry']
    nervous_words = ['nervous', 'anxious', 'worried']
    sad_words = ['sad','gloomy','very sad']
    sick_words = ['sick','fever', 'pain','cold', 'very sick']

    #일기 내용 토큰화
    tokens = word_tokenize(summary.lower())
    # 단어별로 감정 카테고리 확인
    emotion = None
    for token in tokens:
        if token in happy_words:
            emotion = "happy"
            break
        elif token in good_words:
            emotion = "good"
            break
        elif token in neutral_words:
            emotion = "neutral"
            break
        elif token in bad_words:
            emotion = "bad"
            break
        elif token in confused_words:
            emotion = "confused"
            break
        elif token in angry_words:
            emotion = "angry"
            break
        elif token in nervous_words:
            emotion = "nervous"
            break
        elif token in sad_words:
            emotion = "sad"
            break
        elif token in sick_words:
            emotion = "sick"
            break
        if  emotion is None:
            emotion = "neutral"
    print("emotion : " + emotion)
    return emotion
    #return {emotion, summary}

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
                filepath = os.path.join(img_folder, filename)
                img.save(filepath)
                try:
                    s3.upload_file(filepath,"picaboodiaryimage",filename)
                except Exception as e:
                    print(e)
    # 이미지 개수 확인 로직
    image_count = len([name for name in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, name))])
    if image_count > max_image_count:
        # 이미지 파일들을 수정 시간을 기준으로 정렬
        files = sorted(os.listdir(img_folder), key=lambda x:os.path.join(img_folder, x))
        os.remove(os.path.join(img_folder, files[0]))

    return filename