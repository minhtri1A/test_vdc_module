import io
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import torchaudio
from vdc_module.voice import (
    generate_text_to_speech,
    generate_speech_to_text,
    load_model_text_to_speech_vi,
    load_model_speech_to_text,
)
import tempfile
import shutil
from pyngrok import ngrok
import time

# Khởi tạo FastAPI
app = FastAPI(
    title="Vietnamese Text-to-Speech API",
    description=f"API sử dụng model để chuyển văn bản tiếng Việt thành giọng nói.",
    version="1.0.0",
)

# load model text_to_speech vdc_module
load_model_text_to_speech_vi()
load_model_speech_to_text()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()  # bắt đầu đo thời gian
    response = await call_next(request)
    process_time = time.time() - start_time  # tính thời gian xử lý
    response.headers["X-Process-Time"] = str(process_time)  # thêm header
    return response


@app.get("/tts")
async def getTTS(tts_text: str):

    out_wav = generate_text_to_speech(tts_text, normalize_text=True)
    buffer = io.BytesIO()
    torchaudio.save(buffer, out_wav, 24000, format="wav")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"},
    )


@app.post("/stt")
async def getSTT(file: UploadFile = File(...)):
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    text = generate_speech_to_text(tmp_path)

    return JSONResponse(content={"text": text})


ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # Chạy với uvicorn
