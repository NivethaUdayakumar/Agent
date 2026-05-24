import sys
import traceback
from pathlib import Path

model_path = Path(r"C:\Users\nivet\Agent\Model\Qwen3.5-4B-UD-IQ2_XXS.gguf")

print(f"Model path: {model_path}")
print(f"Model exists: {model_path.exists()}")
print(f"Model size: {model_path.stat().st_size / (1024**3):.2f} GB")

try:
    import llama_cpp
    print(f"\nllama_cpp version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'}")
    
    print("\nAttempting to load model...")
    llm = llama_cpp.Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=128,
        n_threads=1,
        verbose=True,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"\nError: {e}")
    traceback.print_exc()
