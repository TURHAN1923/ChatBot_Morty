import os
import whisper
import pyttsx3
import sounddevice as sd
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import whisper
import pyttsx3
import sounddevice as sd
import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

login(token="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")

LORA_PATH = r"C:\Users\emret\OneDrive\Masaüstü\checkpoint-42000"
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
SAVE_PATH = r"C:\Users\emret\OneDrive\Masaüstü\llama2-base"

print("📥 Base model indiriliyor (ilk seferde uzun sürebilir)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Kaydet
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    base_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


print(" LoRA ağırlıkları entegre ediliyor...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def record_audio(duration=5, samplerate=16000):
    print(" listining...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio):
    model_whisper = whisper.load_model("tiny")
    result = model_whisper.transcribe(audio, language="en")
    return result["text"]

def format_prompt(instruction):
    return f"User: {instruction}\nAssistant:"

def query_local_model(instruction):
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,            # Daha kısa cevap
        temperature=0.7,              # Yaratıcılığı düşür
        top_p=0.9,                   # Odaklı örnekleme
        repetition_penalty=1.2,       # Tekrarları azalt
        no_repeat_ngram_size=3,       # 3 kelimelik tekrar engeli
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Prompt'u çıkart ve sadece cevabı döndür
    if prompt in result:
        result = result.split(prompt)[-1].strip()
    return result


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.say(text)
    engine.runAndWait()

def open_application(text):
    text = text.lower()
    if "notepad" in text:
        os.system("notepad.exe")
    elif "calculator" in text:
        os.system("calc.exe")
    elif "chrome" in text:
        os.system("start chrome")
    elif "spotify" in text:
        os.system("start spotify")
    elif "steam" in text:
        os.system("start steam")
    elif "firefox" in text:
        os.system("start firefox")
    elif "league of legends" in text or "lol" in text:
        lol_path = r'"C:\Riot Games\League of Legends\LeagueClient.exe"'
        os.system(f"start {lol_path}")
    else:
        print("No matching application found.")

def main():
    audio = record_audio()
    user_text = transcribe_audio(audio)
    print(f" You said: {user_text}")

    response = query_local_model(user_text)
    print(f"Model says: {response}")

    speak(response)
    open_application(user_text)

if __name__ == "__main__":
    main()
