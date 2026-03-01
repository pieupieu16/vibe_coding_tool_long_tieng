import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print(f"⏳ Đang tải siêu mô hình {MODEL_NAME} vào GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True).to("cuda")
print("✅ Tải mô hình NLLB thành công!")

VIETNAMESE_ID = tokenizer.convert_tokens_to_ids("vie_Latn")

# --- TỪ ĐIỂN CHUYÊN NGÀNH (BẢO HIỂM CHỐNG DỊCH SAI) ---
DOMAIN_TERMS = {
    "artificial intelligence": "Trí tuệ nhân tạo",
    "machine learning": "Học máy",
    "data science": "Khoa học dữ liệu",
    "data preprocessing": "Tiền xử lý dữ liệu",
    "customer churn": "Tỉ lệ rời bỏ của khách hàng"
}

def translate_en_to_vi(text):
    if not text.strip():
        return ""
    
    # Kỹ thuật Code-switching: Ép AI học thuộc từ vựng của mình trước khi dịch
    processed_text = text
    for eng_term, vi_term in DOMAIN_TERMS.items():
        pattern = re.compile(re.escape(eng_term), re.IGNORECASE)
        processed_text = pattern.sub(vi_term, processed_text)
    
    inputs = tokenizer(processed_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=VIETNAMESE_ID,
            max_length=100
        )
    
    result_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result_text

if __name__ == "__main__":
    test_text = "Hello, I am currently learning Artificial Intelligence and Data Science at Thang Long University."
    print(f"\n🇬🇧 Bản gốc: {test_text}")
    vi_text = translate_en_to_vi(test_text)
    print(f"🇻🇳 Bản dịch: {vi_text}\n")