import transformers

import torch

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
max_memory = f"{free_in_GB - 2}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

tokenizer = transformers.LLaMATokenizer.from_pretrained(
    "/home/ec2-user/SageMaker/llama/converted/7B/tokenizer/")
model = transformers.LLaMAForCausalLM.from_pretrained(
    "/home/ec2-user/SageMaker/llama/converted/7B/llama-7b/",
    device_map="auto",
    max_memory=max_memory
)

print("finish loading")

batch = tokenizer(
    "The primary use of LLaMA is research on large language models, including",
    return_tensors="pt",
    add_special_tokens=False
)
batch = {k: v.cuda() for k, v in batch.items()}
with torch.inference_mode():
    generated = model.generate(batch["input_ids"], max_length=100)
    print(tokenizer.decode(generated[0]))
