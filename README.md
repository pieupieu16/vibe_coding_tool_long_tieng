---
title: Valtec Vietnamese TTS
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Valtec Vietnamese TTS - 5-Speaker System

🎙️ **Vietnamese Text-to-Speech with Regional Accents**

Vietnamese TTS system supporting 5 different voices with Northern and Southern regional accents.

## Features

- ✅ **5 Vietnamese Voices**: NF, SF, NM1, SM, NM2 (Northern/Southern, Male/Female)
- ✅ **Accurate G2P**: Vietnamese grapheme-to-phoneme conversion
- ✅ **High Quality**: VITS-based neural TTS
- ✅ **Fast Inference**: GPU-accelerated synthesis
- ✅ **Browser Demo**: ONNX Runtime Web version available

## Available Speakers

| Code | Region | Gender | Description |
|------|--------|--------|-------------|
| **NF** | Northern (Bắc) | Female | Formal, clear pronunciation |
| **SF** | Southern (Nam) | Female | Friendly, casual tone |
| **NM1** | Northern (Bắc) | Male | Professional voice |
| **SM** | Southern (Nam) | Male | Conversational style |
| **NM2** | Northern (Bắc) | Male | Authoritative tone |

## Usage

### Gradio Interface (This Space)

1. Enter Vietnamese text
2. Select a voice
3. Adjust synthesis parameters
4. Click "Generate Speech"

### Web Demo (Static HTML)

Switch to the "Web Demo" tab for browser-based inference using ONNX Runtime Web.

### Python API

```python
from valtec_tts import VietnameseTTS

tts = VietnameseTTS()
audio = tts.synthesize("Xin chào Việt Nam", speaker="NF")
```

## Model Info

- **Architecture**: VITS (Conditional Variational Autoencoder)
- **Speakers**: 5 (Northern/Southern Vietnamese accents)
- **Sample Rate**: 24kHz
- **Model Size**: ~220MB (PyTorch), ~165MB (ONNX)

## Links

- 🏠 [GitHub Repository](https://github.com/valtecAI-team/valtec-tts)
- 📦 [ONNX Models](https://huggingface.co/valtecAI-team/valtec-tts-onnx)
- 🎯 [Edge Deployment Guide](https://github.com/valtecAI-team/valtec-tts/tree/main/deployments/edge)

## License

MIT License - Free for commercial and non-commercial use.

---

**Powered by Valtec AI Team** | Built with Gradio & ONNX Runtime
