import uvicorn
from fastapi import FastAPI
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel,PeftConfig
import sys
import time

MODEL_PATH="../models_hf/7B"
PORT = 5050

finetune_path = sys.argv[1]
PORT = sys.argv[2]

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model =LlamaForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model,finetune_path)

app = FastAPI()

@app.post("/")
async def sentence_split(info:dict):
    print(info['command'])
    command = info['command']


    # eval_prompt = f"""
    # what is the verbs, direct objects, and indirect objects of this sentence:
    # {command}
    # ---
    # [verb,[direct objects],indirect object]
    # """

    model_input = tokenizer(command, return_tensors="pt").to("cuda")

    
    with torch.no_grad():
        result = tokenizer.decode(model.generate(**model_input, max_new_tokens=50)[0], skip_special_tokens=True)
    
    result = result.split("\n")[-1].strip()

    print(result)
    return result

    
    
if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=PORT)