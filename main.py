import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORTED
from google import genai
from google.genai import types
import io
# Initialize the App
app = FastAPI(title="Trex AI Image Generator")
# --- CORS SETUP (NEW) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY is not set. The app will fail to generate images.")
client = genai.Client(api_key=api_key)
@app.get("/")
def home():
    return {"status": "online", "message": "Trex AI Image Generator is Running with Gemini 2.5 Flash Image"}
@app.post("/generate-image")
async def generate_image(prompt: str = Body(..., embed=True)):
    """
    Generates an image using 'gemini-2.5-flash-image'.
    Returns the image directly as a PNG stream.
    """
    try:
        print(f"Generating image for prompt: {prompt}")
        
        # Call Gemini 2.5 Flash Image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="image/png"
            )
        )
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
        
        raise HTTPException(status_code=500, detail="Model returned a response, but no image data was found.")
    except Exception as e:
        print(f"Error generating image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)