from llama_cpp import Llama

# 初始化 Llama 模型（只初始化一次）
llm = Llama(
    model_path=r"C:\Users\a0931\OneDrive\桌面\大專生\web\phi-3-mini-128k-instruct.Q4_K_M.gguf",
    n_ctx=4096,          # 上下文長度，phi-3 建議設 2048~4096
    n_threads=8,         # 根據你的 CPU 核心數調整
    n_gpu_layers=0       # 0 表示純 CPU；也可以設 20 等 GPU 加速
)

# 封裝成函式供外部呼叫
def generate_response(messages):
    
    gresponse = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        stop=["</s>"]
    )
    return gresponse["choices"][0]["message"]["content"].strip()


