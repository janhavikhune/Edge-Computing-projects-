from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

print("🔁 Loading TinyLlama model (FP16 optimized)... please wait")

# ✅ Model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ✅ Load tokenizer and model with FP16 support if CUDA is available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"   # Automatically assigns GPU/CPU layers
)

print("✅ Agentic AI model loaded successfully (FP16 GPU enabled)!")

# --------- API Endpoint ----------
@app.post("/query")
async def query_agent(request: Request):
    """Receives YOLO detections and returns AI decision"""
    data = await request.json()
    detection = data.get("detection", "")
    confidence = data.get("confidence", 0)
    prev_action = data.get("prev_action", "None")

    prompt = (
        f"You are a security AI assistant. A {detection} was detected "
        f"with {confidence}% confidence. Previous action: {prev_action}. "
        f"What should be done next? Choose from: (A) Log silently, "
        f"(B) Trigger LED alert, (C) Sound buzzer, (D) Notify owner."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=60)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n🤖 Agentic AI Decision:\n{reply}\n")
    return {"response": reply}
