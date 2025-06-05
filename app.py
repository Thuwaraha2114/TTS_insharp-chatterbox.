# app.py
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import io
import torchaudio
import uvicorn

from pathlib import Path
from services.chatterboxService import ChatterboxService
from services.chatterbox.tts import ChatterboxTTS

app = FastAPI()

# Load model once during startup
base_dir = Path(__file__).resolve().parent
audio_prompt_path = base_dir / "services" / "thuwa.wav"

# Load model only once ✅
chatterbox_model = ChatterboxTTS.from_pretrained(device="cpu")
tts_service = ChatterboxService(model=chatterbox_model, audio_prompt_path=audio_prompt_path)

# Request schema
class TextRequest(BaseModel):
    text: str

@app.post("/generate-audio")
def generate_audio(input_text: TextRequest):
    audio_tensor, sample_rate = tts_service.convert_text_to_voice(input_text.text)

    # Check shape and fix if necessary
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format="wav")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output.wav"
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
