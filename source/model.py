import os
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from dotenv import load_dotenv
#from accelerate.utils import set_seed 
#from accelerate.utils import BnbQuantizationConfig


#set_seed(42)

load_dotenv()

DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
)


class Model: 
    def __init__(self, model_id: str = "facebook/blenderbot-400M-distill", device = 'mps'): 
        
        ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                       cache_dir=DIR)
        self.quant = BitsAndBytesConfig(load_in_4bit = True, 
                                           bnb_4bit_compute_dtype = torch.bfloat16)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                                                       cache_dir=DIR, 
                                                                       quantization_config=self.quant, 
                                                                       device_map = "auto", 
                                                                       token = ACCESS_TOKEN)
        
        self.model.eval()
        self.chat = []
        self.device = device 
        
    def generate(self, question: str, context: str, max_new_tokens: int = 250): 
        
        if context == None or context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""Give a detailed answer to the following question. Question: {question} Context: {context}"""
            
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, 
            tokenize = False, 
            add_generation_prompt = True, 
        ) 
        
        print(formatted_prompt)
        inputs = self.tokenizer.encode(
            formatted_prompt, 
            add_special_tokens = False,
            return_tensors = "pt").to(self.device)
        
        with torch.inference_mode(): 
            outputs = self.model.generate(
                input_ids = inputs,
                max_new_tokens = max_new_tokens,
                do_sample = False, 
            ) 
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        response = response[len(formatted_prompt):].strip()
        response = response.replace("<eos>", "").strip()
        
        return response
        