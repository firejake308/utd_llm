from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_name_or_path = "TheBloke/Llama-2-7B-chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q2_K.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
llm = Llama(model_path=model_path, n_threads=8, n_batch=512, n_ctx=2048)

app = FastAPI()

@app.get("/")
async def root():
    return "Welcome to ToThePoint backend"

@app.get("/completion")
async def completion():
    prompt="""[INST] <<SYS>>
You are a helpful and honest research assistant. Use the context below to answer any questions you are asked. At the end of the answer, list relevant quotes from the context. If the answer is not found in the context, then answer to the best of your ability, but do not share any references. Format your answers as follows:

This is an example answer, with an explanation that ties everything together.
RELEVANT QUOTES:
* "This is a quote from the context that supports the answer" -Source Title

With that format in mind, here is the context:

Context:
Whales And Dolphins:
"Whales and dolphins are mammals that live in the sea. Most mammals do not live in the sea, but whales and dolphins are exceptions. This is because they are good swimmers."
----
<</SYS>>


Where do dolphins live? [/INST]"""
    response = llm(
        prompt=prompt,
        max_tokens=2048,
        temperature=0.5,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=1,
        echo=True
        )

    return response["choices"][0]["text"]
