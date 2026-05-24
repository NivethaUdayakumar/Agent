from ctransformers import AutoModelForCausalLM

model_path = r"C:\Users\nivet\Agent\Model\gemma3-1b-Q8_0.gguf"

# Try different model types
for model_type in ["auto", "gemma", "mistral", "llama"]:
    try:
        print(f"Trying model_type='{model_type}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            gpu_layers=0,
        )
        print(f"  ✓ Success with '{model_type}'!")
        break
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
