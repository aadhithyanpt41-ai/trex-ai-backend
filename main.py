import os
import io
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# ===============================
# APP INITIALIZATION
# ===============================
app = FastAPI(title="Trex AI Image Generator")

# ===============================
# CORS (ALLOW NETLIFY / ANY FRONTEND)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# GEMINI CLIENT
# ===============================
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=API_KEY)

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def home():
    return {
        "status": "online",
        "model": "gemini-2.5-flash-image",
        "message": "Trex AI Image Generator is running 🦖🔥"
    }

# ===============================
# IMAGE GENERATION ENDPOINT
# ===============================
@app.post("/generate-image")
async def generate_image(prompt: str = Body(..., embed=True)):
    try:
        print("🖼️ Prompt:", prompt)

        # ---- CORRECT IMAGE API ----
        result = client.models.generate_image(
            model="gemini-2.5-flash-image",
            prompt=prompt
        )

        if not result.images:
            raise HTTPException(status_code=500, detail="No image returned from Gemini")

        image_bytes = result.images[0].data

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png"
        )

    except Exception as e:
        print("🔥 ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# RENDER ENTRY POINT
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
