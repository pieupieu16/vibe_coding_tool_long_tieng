#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vietnamese TTS - Hugging Face Spaces Demo
Giao diện web chuyển văn bản tiếng Việt thành giọng nói.
"""

import os
import sys
import json
import tempfile
import glob
import re
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import gradio as gr

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.vietnamese.text_processor import process_vietnamese_text
from src.vietnamese.phonemizer import text_to_phonemes, VIPHONEME_AVAILABLE
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols
from src.utils import helpers as utils


# ============== Viphoneme Check ==============

def check_viphoneme():
    """Check if viphoneme is working properly."""
    from src.vietnamese.phonemizer import VIPHONEME_AVAILABLE, text_to_phonemes
    
    print(f"Viphoneme available: {VIPHONEME_AVAILABLE}")
    
    if VIPHONEME_AVAILABLE:
        try:
            # Test with a simple Vietnamese text
            test_text = "Xin chào"
            phones, tones, word2ph = text_to_phonemes(test_text, use_viphoneme=True)
            print(f"✅ Viphoneme test passed: '{test_text}' -> {phones[:5]}...")
            return True
        except Exception as e:
            print(f"❌ Viphoneme test failed: {e}")
            return False
    else:
        print("⚠️ Viphoneme not available, using fallback phonemizer")
        return False


# ============== Model Loading ==============

def find_latest_checkpoint(model_dir, prefix="G"):
    """Find the latest checkpoint in model directory."""
    pattern = os.path.join(model_dir, f"{prefix}*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    def get_step(path):
        match = re.search(rf'{prefix}(\d+)\.pth', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def download_model():
    """Download model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    
    hf_repo = "valtecAI-team/valtec-tts-pretrained"
    
    # Get cache directory
    if os.name == 'nt':  # Windows
        cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:  # Linux/Mac
        cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    
    model_dir = cache_base / 'valtec_tts' / 'models' / 'vits-vietnamese'
    
    # Check if already downloaded
    config_path = model_dir / "config.json"
    if config_path.exists():
        checkpoints = list(model_dir.glob("G_*.pth"))
        if checkpoints:
            print(f"Using cached model from: {model_dir}")
            return str(model_dir)
    
    print(f"Downloading model from {hf_repo}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_repo, local_dir=str(model_dir))
    print("Download complete!")
    
    return str(model_dir)


class VietnameseTTS:
    """Vietnamese TTS synthesizer using trained VITS-based model."""
    
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        self.device = device
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.sampling_rate = self.config['data']['sampling_rate']
        self.spk2id = self.config['data']['spk2id']
        self.speakers = list(self.spk2id.keys())
        self.add_blank = self.config['data'].get('add_blank', True)
        
        print(f"Available speakers: {self.speakers}")
        
        # Load model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path):
        """Load the trained model."""
        self.model = SynthesizerTrn(
            len(symbols),
            self.config['data']['filter_length'] // 2 + 1,
            self.config['train']['segment_size'] // self.config['data']['hop_length'],
            n_speakers=self.config['data']['n_speakers'],
            **self.config['model'],
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DDP checkpoint
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def text_to_sequence(self, text, speaker):
        """Convert text to model input tensors."""
        from src.text import cleaned_text_to_sequence
        from src.nn import commons
        
        # Normalize text
        normalized_text = process_vietnamese_text(text)
        
        # Convert to phonemes
        phones, tones, word2ph = text_to_phonemes(normalized_text, use_viphoneme=VIPHONEME_AVAILABLE)
        
        # Convert to sequence
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")
        
        # Add blanks if needed
        if self.add_blank:
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            lang_ids = commons.intersperse(lang_ids, 0)
        
        # Get speaker ID
        if speaker not in self.spk2id:
            print(f"Warning: Speaker '{speaker}' not found, using first speaker: {self.speakers[0]}")
            speaker = self.speakers[0]
        speaker_id = self.spk2id[speaker]
        
        # Create tensors
        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        language = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)
        
        # Create dummy BERT features
        bert = torch.zeros(1024, len(phone_ids)).unsqueeze(0).to(self.device)
        ja_bert = torch.zeros(768, len(phone_ids)).unsqueeze(0).to(self.device)
        
        return x, x_lengths, tone, language, sid, bert, ja_bert
    
    @torch.no_grad()
    def synthesize(self, text, speaker, sdp_ratio=0.0, noise_scale=0.667, 
                   noise_scale_w=0.8, length_scale=1.0):
        """Synthesize speech from text."""
        x, x_lengths, tone, language, sid, bert, ja_bert = self.text_to_sequence(text, speaker)
        
        audio, attn, *_ = self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        
        audio = audio[0, 0].cpu().numpy()
        return audio, self.sampling_rate


# ============== Gradio Interface ==============

class TTSInterface:
    """Wrapper for TTS model with Gradio interface."""
    
    def __init__(self):
        print("Initializing TTS...")
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Download/load model
        model_dir = download_model()
        checkpoint_path = find_latest_checkpoint(model_dir, "G")
        config_path = os.path.join(model_dir, "config.json")
        
        self.tts = VietnameseTTS(checkpoint_path, config_path, self.device)
        self.temp_dir = Path(tempfile.gettempdir()) / "valtec_tts_demo"
        self.temp_dir.mkdir(exist_ok=True)
        
        print("TTS initialized successfully!")
    
    def synthesize(self, text, speaker, speed, noise_scale, noise_scale_w, sdp_ratio):
        """Synthesize speech from text."""
        try:
            if not text or not text.strip():
                return None, "⚠️ Vui lòng nhập văn bản"
            
            audio, sr = self.tts.synthesize(
                text=text.strip(),
                speaker=speaker,
                length_scale=speed,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                sdp_ratio=sdp_ratio,
            )
            
            # Save to temp file
            output_path = self.temp_dir / f"output_{hash(text)}.wav"
            sf.write(str(output_path), audio, sr)
            
            return str(output_path), f"✅ Tạo giọng nói thành công! ({len(audio)/sr:.2f}s)"
            
        except Exception as e:
            return None, f"❌ Lỗi: {str(e)}"


def create_demo(tts_interface):
    """Create Gradio interface."""
    
    examples = [
        ["Xin chào, chúc bạn một ngày tốt lành", "male", 1.0, 0.667, 0.8, 0.0],
        ["Buổi sáng hôm nay trời trong xanh và gió thổi rất nhẹ", "male", 1.0, 0.667, 0.8, 0.0],
        ["Tôi pha một tách cà phê nóng và ngồi nhìn ánh nắng chiếu qua cửa sổ", "female", 1.0, 0.667, 0.8, 0.0],
        ["Việt Nam là một đất nước xinh đẹp với văn hóa phong phú", "male", 0.9, 0.667, 0.8, 0.0],
        ["Con đường làng quê rợp bóng tre xanh mát rượi", "female", 1.1, 0.667, 0.8, 0.0],
    ]
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
        title="Chuyển Văn Bản Thành Giọng Nói Tiếng Việt",
        css="""
        /* Base styles - Mobile first */
        .gradio-container { 
            max-width: 100% !important; 
            margin: 0 auto !important;
            padding: 10px !important;
        }
        .main {
            margin: 0 auto !important;
            padding: 0 10px !important;
        }
        #title {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 1.5rem;
        }
        .contain {
            max-width: 100% !important;
            margin: 0 auto !important;
        }
        
        /* Tablet (768px+) */
        @media (min-width: 768px) {
            .gradio-container { 
                max-width: 750px !important; 
                padding: 20px !important;
            }
            #title {
                font-size: 2rem;
            }
        }
        
        /* Desktop (1024px+) */
        @media (min-width: 1024px) {
            .gradio-container { 
                max-width: 900px !important; 
            }
            #title {
                font-size: 2.5rem;
            }
        }
        
        /* Large screens (1200px+) */
        @media (min-width: 1200px) {
            .gradio-container { 
                max-width: 1000px !important; 
            }
        }
        """
    ) as demo:
        
        gr.Markdown("""
            # <span id="title">🎙️ Chuyển Văn Bản Thành Giọng Nói</span>
            
            ### Hệ thống tổng hợp giọng nói tiếng Việt
            
            Nhập văn bản tiếng Việt và chọn giọng đọc để tạo audio.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="📝 Văn bản đầu vào",
                    placeholder="Nhập văn bản tiếng Việt ở đây...",
                    lines=5,
                    max_lines=10,
                )
                
                speaker_dropdown = gr.Dropdown(
                    choices=tts_interface.tts.speakers,
                    value=tts_interface.tts.speakers[0],
                    label="🎤 Chọn giọng đọc",
                    info="Chọn người đọc từ danh sách"
                )
                
                synthesize_btn = gr.Button(
                    "🔊 Tạo giọng nói",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    speed_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Tốc độ",
                        info="< 1.0: Nhanh hơn | > 1.0: Chậm hơn"
                    )
                    
                    noise_scale_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.667, step=0.01,
                        label="Noise Scale",
                        info="Điều khiển độ biến thiên giọng nói"
                    )
                    
                    noise_scale_w_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.8, step=0.01,
                        label="Duration Noise",
                        info="Điều khiển độ biến thiên thời lượng"
                    )
                    
                    sdp_ratio_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                        label="SDP Ratio",
                        info="0: Xác định | 1: Ngẫu nhiên"
                    )
        
        with gr.Row():
            with gr.Column():
                audio_output = gr.Audio(
                    label="🔊 Audio đầu ra",
                    type="filepath",
                    interactive=False
                )
                status_output = gr.Textbox(
                    label="📊 Trạng thái",
                    interactive=False,
                    show_label=False
                )
        
        gr.Markdown("### 📚 Ví dụ")
        gr.Examples(
            examples=examples,
            inputs=[text_input, speaker_dropdown, speed_slider, 
                    noise_scale_slider, noise_scale_w_slider, sdp_ratio_slider],
            outputs=[audio_output, status_output],
            fn=tts_interface.synthesize,
            cache_examples=False,
        )
        
        synthesize_btn.click(
            fn=tts_interface.synthesize,
            inputs=[text_input, speaker_dropdown, speed_slider,
                    noise_scale_slider, noise_scale_w_slider, sdp_ratio_slider],
            outputs=[audio_output, status_output],
        )
        
        gr.Markdown("""
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                Hệ thống tổng hợp giọng nói tiếng Việt | Powered by <b>Valtec AI Team</b>
            </div>
        """)
    
    return demo


# ============== Main ==============

if __name__ == "__main__":
    print("Đang khởi động hệ thống tổng hợp giọng nói tiếng Việt...")
    
    # Check viphoneme
    check_viphoneme()
    
    tts_interface = TTSInterface()
    demo = create_demo(tts_interface)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
