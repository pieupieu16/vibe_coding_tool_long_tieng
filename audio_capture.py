import sounddevice as sd
import numpy as np

RATE = 16000

def get_virtual_cable_input_id():
    """Tự động tìm ID của CABLE Output một cách an toàn"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        # Kiểm tra thiết bị có hỗ trợ đầu vào (max_input_channels > 0)
        if 'CABLE Output' in dev['name'] and dev['max_input_channels'] > 0:
            return i
    return None

def start_audio_stream(audio_queue):
    device_id = get_virtual_cable_input_id()
    
    if device_id is None:
        print("❌ LỖI: Không tìm thấy CABLE Output. Hãy kiểm tra lại Sound Settings!")
        return

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ Status âm thanh: {status}")
        
        # Chuyển đổi an toàn từ bộ đệm Raw sang mảng NumPy
        audio_chunk = np.frombuffer(indata, dtype='int16').copy()
        audio_queue.put(audio_chunk)

    print(f"🎙️ AI đang lắng nghe Video qua ID {device_id}...")
    
    try:
        # Sử dụng InputStream thay vì RawInputStream để Windows tự điều phối tốt hơn
        with sd.InputStream(samplerate=RATE, blocksize=2048, 
                             device=device_id, dtype='int16', 
                             channels=1, callback=callback):
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"🛑 Lỗi khởi tạo âm thanh: {e}")