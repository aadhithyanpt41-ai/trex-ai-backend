import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from google import genai
from google.genai import types
import io

# Initialize the App
app = FastAPI(title="Trex AI Image Generator")

# Initialize Gemini Client
# We get the API Key from the Environment Variable set in Render
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
    Generates an image using 'gemini-2.5-flash-image' (Nano Banana).
    Returns the image directly as a PNG stream.
    """
    try:
        print(f"Generating image for prompt: {prompt}")
        
        # Call Gemini 2.5 Flash Image
        # Note: We use generate_content for this model, NOT generate_images
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="image/png"
            )
        )

        # Extract the image bytes from the response
        # The image is usually returned as 'inline_data' inside the first part
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
        
        # If no image found in response
        raise HTTPException(status_code=500, detail="Model returned a response, but no image data was found.")

    except Exception as e:
        print(f"Error generating image: {e}")
        # Return the error message so you can debug it in your frontend
        return JSONResponse(status_code=500, content={"error": str(e)})

# Start the server (Required for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)