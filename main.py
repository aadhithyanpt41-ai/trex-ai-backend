import os
import io
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# ===============================
# APP INIT
# ===============================
app = FastAPI(title="Trex AI Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# GEMINI CONFIG
# ===============================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")  # chat model

# ===============================
# HEALTH
# ===============================
@app.get("/")
def home():
    return {"status": "online"}

# ===============================
# IMAGE GENERATION
# ===============================
@app.post("/generate-image")
async def generate_image(prompt: str = Body(..., embed=True)):
    try:
        print("Generating image for:", prompt)

        response = model.generate_content(
            f"Generate an image: {prompt}",
            generation_config={"temperature": 0.4}
        )

        # ⚠️ Gemini via this SDK DOES NOT return raw PNG
        # So we must return TEXT fallback
        return {
            "warning": "Gemini free SDK does not return raw images",
            "output": response.text
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# START
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
