from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    max_tokens=200,
    temperature=0,
    verbose=False
)

response = llm.invoke("Find the issue: ERROR: placement legalization failed")

print(response)
