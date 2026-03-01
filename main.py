import queue
import threading
import time
import numpy as np
from faster_whisper import WhisperModel

# Import các module
from audio_capture import start_audio_stream
from streaming_translation import translate_en_to_vi
from tts_valtec import speak_vi

# Hàng đợi trung chuyển dữ liệu
audio_queue = queue.Queue()
text_en_queue = queue.Queue()
text_vi_queue = queue.Queue()

print("⏳ Đang khởi động Whisper Engine trên GPU...")
stt_model = WhisperModel("tiny.en", device="cuda", compute_type="int8")

def stt_worker():
    audio_buffer = []
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            audio_buffer.append(data)
            
            # Gom đủ ~2 giây âm thanh để dịch đúng ngữ cảnh
            if len(audio_buffer) > 45:
                full_audio = np.concatenate(audio_buffer)
                audio_fp32 = full_audio.astype(np.float32) / 32768.0
                
                # Ép Whisper chỉ nghe tiếng Anh
                segments, _ = stt_model.transcribe(audio_fp32, language="en", beam_size=5)
                
                for segment in segments:
                    if segment.text.strip():
                        print(f"🇬🇧 Nghe được: {segment.text.strip()}")
                        text_en_queue.put(segment.text.strip())
                
                audio_buffer = [] 
        time.sleep(0.01)
        

def translation_worker():
    while True:
        if not text_en_queue.empty():
            eng_text = text_en_queue.get()
            print(f"🔄 Đang gửi sang NLLB để dịch: {eng_text}") # Dòng kiểm tra 1
            
            try:
                vi_text = translate_en_to_vi(eng_text)
                if vi_text:
                    print(f"✅ NLLB trả về: {vi_text}") # Dòng kiểm tra 2
                    text_vi_queue.put(vi_text)
                    
                else:
                    print("⚠️ Cảnh báo: Hàm dịch trả về kết quả rỗng!")
            except Exception as e:
                print(f"❌ Lỗi tại luồng Dịch: {e}")
        time.sleep(0.01)
def tts_worker():
    """Luồng 3: Đảm bảo phát âm thanh theo thứ tự, không chồng lấn"""
    while True:
        try:
            # Lấy câu tiếng Việt tiếp theo từ hàng đợi
            if not text_vi_queue.empty():
                vi_text = text_vi_queue.get()
                
                # Hàm speak_vi bên trong phải có sd.wait() hoặc cơ chế đợi phát xong
                speak_vi(vi_text)
                
                # Nghỉ một chút để hệ thống giải phóng bộ nhớ sau mỗi câu
                time.sleep(0.2) 
        except Exception as e:
            print(f"⚠️ Lỗi luồng TTS: {e}")
        time.sleep(0.1)

if __name__ == "__main__":
    t1 = threading.Thread(target=start_audio_stream, args=(audio_queue,), daemon=True)
    t2 = threading.Thread(target=stt_worker, daemon=True)
    t3 = threading.Thread(target=translation_worker, daemon=True)
    t4 = threading.Thread(target=tts_worker, daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    print("\n🚀 HỆ THỐNG LỒNG TIẾNG ĐÃ SẴN SÀNG!")
    print("👉 Hãy mở Video và đảm bảo âm lượng Youtube ở mức 10%.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Đang tắt hệ thống...")