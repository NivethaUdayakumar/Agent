from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"C:\Users\nivet\Agent\Model\gemma3-1b-Q8_0.gguf"

print("Testing transformers with GGUF...")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b", trust_remote_code=True)
    
    print("Loading model from GGUF...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        load_in_8bit=False,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    print("✓ Model loaded successfully!")
    
    # Test inference
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
