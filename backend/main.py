from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

from services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl
    # advanced settings

app = FastAPI()

# allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# config AI
genai_processor = GeminiProcessor(
    model_name="gemini-pro", 
    project="ai-workspace-425101"
)

@app.post("/analyze_video/")
def analyze_video(request: VideoAnalysisRequest):
    doc_processor = YoutubeProcessor(genai=genai_processor)
    results = doc_processor.retrieve_youtube_documents(str(request.youtube_link), verbose=True)
    
    # summary = genai_processor.generate_document_summary(results, verbose=True)

    key_concepts = doc_processor.find_key_concepts(results, sample_size=10, verbose=True)
    
    return {
        "key_concepts": key_concepts,
    }
    
@app.get("/")
def health():
    return {"status": "ok"}