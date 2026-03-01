import queue
import threading
import numpy as np
from faster_whisper import WhisperModel
from audio_capture import start_audio_stream, RATE

# 1. Khởi tạo Hàng đợi (Queue) để giao tiếp giữa Mic và AI
audio_queue = queue.Queue()

# 2. Khởi tạo mô hình AI
# Với sức mạnh từ card đồ họa rời trên chiếc Lenovo LOQ 2023, 
# chúng ta sẽ ép mô hình chạy trên "cuda" (GPU) và dùng "float16" để tăng tốc tối đa.
print("⏳ Đang tải mô hình AI vào GPU...")
model_size = "tiny.en" # Dùng bản tiếng Anh nhỏ nhất để test tốc độ
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("✅ Tải mô hình thành công!")

def process_audio():
    """Hàm này liên tục lấy âm thanh từ Queue để dịch thành chữ"""
    buffer = []
    chunk_count = 0
    
    # Tính toán: Chúng ta cần gom khoảng 1.5 giây âm thanh mỗi lần xử lý
    # Công thức: (16000 mẫu/giây * 1.5 giây) / 1024 mẫu/chunk ≈ 23 chunks
    required_chunks = int((RATE * 1.5) / 1024) 
    
    while True:
        # Rút 1 chunk âm thanh từ Queue
        data = audio_queue.get()
        buffer.append(data)
        chunk_count += 1
        
        # Khi gom đủ 1.5 giây âm thanh
        if chunk_count >= required_chunks:
            # Nối các mảnh nhỏ lại thành 1 mảng dài
            audio_data = np.concatenate(buffer)
            
            # Whisper yêu cầu dữ liệu chuẩn float32 (-1.0 đến 1.0)
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            
            # Bắt đầu AI nhận diện (Inference)
            segments, info = model.transcribe(audio_float32, beam_size=5)
            
            for segment in segments:
                if segment.text.strip():
                    print(f"🗣️ AI nghe được: {segment.text}")
            
            # Xóa bộ đệm để bắt đầu gom 1.5 giây tiếp theo
            buffer = []
            chunk_count = 0

# 3. Tạo một Luồng (Thread) riêng biệt chỉ để chạy Microphone
capture_thread = threading.Thread(target=start_audio_stream, args=(audio_queue,))
capture_thread.daemon = True # Thread này sẽ tự tắt khi chương trình chính tắt
capture_thread.start()

# 4. Luồng chính (Main Thread) sẽ chạy AI nhận diện
try:
    process_audio()
except KeyboardInterrupt:
    print("\n🛑 Đã dừng hệ thống STT.")