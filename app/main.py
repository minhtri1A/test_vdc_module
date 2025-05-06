import io
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import torchaudio
from transformers import (
    AutoProcessor,
    VitsModel,
)  # Hoặc MmsForConditionalGeneration, MmsTokenizer tùy vào cách load model MMS
from vdc_module.voice import generate_text_to_speech, generate_speech_to_text
import tempfile
import shutil

# --- Cấu hình và Tải Model (Nên thực hiện một lần khi ứng dụng khởi động) ---
MODEL_ID = "facebook/mms-tts-vie"  # Model ID cho tiếng Việt
SAMPLING_RATE = 16000  # Tần số lấy mẫu chuẩn của model MMS-TTS

# Chọn thiết bị (GPU nếu có, nếu không thì CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Sử dụng thiết bị: {device}")

# Khởi tạo FastAPI
app = FastAPI(
    title="Vietnamese Text-to-Speech API",
    description=f"API sử dụng model {MODEL_ID} để chuyển văn bản tiếng Việt thành giọng nói.",
    version="1.0.0",
)


# --- Định nghĩa Request Body ---
class TextToSpeechRequest(BaseModel):
    text: str
    # Có thể thêm tham số như speed, voice_id nếu model hỗ trợ


# --- Định nghĩa Global Variables ---
processor = None
model = None


# --- Tải Mô Hình khi khởi động ứng dụng ---
def load_model():
    global processor, model
    try:
        print(f"Đang tải processor và model: {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = VitsModel.from_pretrained(MODEL_ID).to(device)
        print("Tải model thành công!")

        # Kiểm tra và lấy tần số lấy mẫu từ model nếu có
        if hasattr(model.config, "sampling_rate"):
            global SAMPLING_RATE
            SAMPLING_RATE = model.config.sampling_rate
            print(f"Sử dụng sampling rate từ model config: {SAMPLING_RATE} Hz")
        else:
            print(
                f"Không tìm thấy sampling_rate trong config, sử dụng mặc định: {SAMPLING_RATE} Hz"
            )
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        processor = None
        model = None


# Gọi hàm load_model() khi ứng dụng khởi động
load_model()


# --- API Endpoint để kiểm tra sức khỏe ---
@app.get("/health")
async def health_check():
    """Kiểm tra xem API có đang chạy và model đã tải chưa."""
    if model and processor:
        return {"status": "ok", "model_loaded": True, "device": device}
    else:
        return {
            "status": "error",
            "model_loaded": False,
            "message": "Model TTS chưa được tải.",
        }


# --- API Endpoint để tổng hợp giọng nói từ văn bản ---
@app.post(
    "/synthesize/",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Trả về file audio WAV.",
        },
        400: {"description": "Input không hợp lệ."},
        500: {"description": "Lỗi server hoặc không thể tạo audio."},
        503: {"description": "Model chưa được tải thành công."},
    },
)
async def synthesize_speech(request: TextToSpeechRequest):
    """Nhận văn bản tiếng Việt và trả về file audio WAV."""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model TTS chưa sẵn sàng.")

    if not request.text:
        raise HTTPException(
            status_code=400, detail="Trường 'text' không được để trống."
        )

    print(f"Nhận yêu cầu tổng hợp cho văn bản: '{request.text}'")

    try:
        # 1. Chuẩn bị input
        inputs = processor(text=request.text, return_tensors="pt").to(device)

        # 2. Sinh waveform âm thanh từ mô hình
        with torch.no_grad():
            speech_waveform = model(**inputs).waveform

        # Đảm bảo waveform ở dạng float32 trên CPU và là numpy array
        speech_np = speech_waveform.squeeze().cpu().float().numpy()
        print("Waveform min:", speech_np.min(), "max:", speech_np.max())
        print("Waveform sum:", speech_np.sum())
        print("Waveform shape:", speech_np.shape)
        print(f"Đã tạo waveform có shape: {speech_np.shape}, dtype: {speech_np.dtype}")

        # 3. Chuyển numpy array thành file WAV trong bộ nhớ
        buffer = io.BytesIO()
        sf.write(
            buffer, speech_np, SAMPLING_RATE, format="WAV", subtype="PCM_16"
        )  # subtype PCM_16 là chuẩn WAV 16-bit
        buffer.seek(0)  # Đưa con trỏ về đầu buffer để đọc
        sf.write("test.wav", speech_np, SAMPLING_RATE, format="WAV", subtype="PCM_16")
        # 4. Trả về StreamingResponse
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        print(f"Lỗi trong quá trình tổng hợp: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể tạo audio: {str(e)}")


# tts_text = "Xin chào, tôi là vua dụng cụ AI, được viết bởi vuadungcu.com, Bạn có thể hỏi tôi bất cứ thứ gì trên đời này, trả lời được hay không thì hên xui."


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


# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # Chạy với uvicorn
