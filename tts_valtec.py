import numpy as np
import scipy.io.wavfile as wavfile
import os
import sys
import subprocess

# Cấu hình đường dẫn cho các module con
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, "src"))

from valtec_tts.tts import TTS

print("⏳ Đang khởi động Voice Engine (Subprocess Mode)...")
tts = TTS()
import sounddevice as sd

def speak_vi(text):
    if not text.strip(): return
    
    audio_data, sample_rate = tts.synthesize(text, speaker="NF")
    
    # Tự động tìm ID của Loa thật (Realtek Audio)
    devices = sd.query_devices()
    speaker_id = None
    for i, dev in enumerate(devices):
        # Chọn thiết bị có 'Realtek' hoặc 'Speakers' và KHÔNG PHẢI 'CABLE'
        if ('Realtek' in dev['name'] or 'Speakers' in dev['name']) \
            and dev['max_output_channels'] > 0 \
            and 'CABLE' not in dev['name']:
            speaker_id = i
            break
            
    if speaker_id is not None:
        print(f"🔊 Đang phát ra loa thật (ID {speaker_id}): {text}")
        sd.play(audio_data, sample_rate, device=speaker_id)
        sd.wait()
    else:
        sd.play(audio_data, sample_rate)
        sd.wait()