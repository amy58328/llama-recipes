import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel,PeftConfig


MODEL_PATH="../llama/models_hf/7B"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model =LlamaForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

use_peft = input("use PEFT, YES:1, NO:2:")
if use_peft == "1":
    model = PeftModel.from_pretrained(model,"../PEFT/model-20240504")

while True:
    question = input("Please enter the command: ")
    eval_prompt = f"""
    what is the verbs, direct objects, and indirect objects of this sentence:
    {question}
    ---
    [verb,[direct objects],indirect object]
    """


    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    
    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=50)[0], skip_special_tokens=True))
    