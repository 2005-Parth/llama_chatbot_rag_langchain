import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if torch.cuda.is_available():
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

def chat_with_llama(prompt):
    print("Prompt:", prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

while True:
    prompt = input("You: ")
    response = chat_with_llama(prompt)
    print("Llama:", response)