from .base import Model
import torch
import logging
import time
from typing import List, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator


class HFLocalModel(Model):
    SUPPORTED_DTYPES = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }
    def __init__(self, model_name: str, 
                    instruct: bool,
                    dtype: str="bfloat16", 
                    max_new_tokens: Optional[int]=None, 
                    temperature: int = 0.1,
                    top_p: float = 0.5
            ):
        assert dtype in self.SUPPORTED_DTYPES, f"{dtype} not supported"
        self.instruct = instruct
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        accelerator = Accelerator() # run accelerate config to use mutli-GPUs

        logging.info(f"[HFLocalModel] Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", # uses pipeline parallelism
            dtype=self.SUPPORTED_DTYPES[dtype],
        )

        if accelerator.is_main_process:
            # Print model specifications
            config = AutoConfig.from_pretrained(model_name)
            logging.info("\n\n[Model specifications] >>> ")
            logging.info(f"Number of layers: {config.num_hidden_layers}")
            logging.info(f"Hidden size: {config.hidden_size}")
            logging.info(f"Vocabulary size: {config.vocab_size}")
            logging.info(f"Number of attention heads: {config.num_attention_heads}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            logging.warning("Tokenizer doesn't have any pad token. Using eos_token instead.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        logging.info(f"[HFLocalModel] {model_name} loaded succesfully!")
    
    @torch.inference_mode()
    def predict(self, prompts: Union[str, List[str]], 
                    batch_size: int=8, 
                    max_new_tokens: Optional[int]=None, 
                    temperature: int = 0.1,
                    top_p: float = 0.5,
        ):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        outputs: List[str] = []

        # use appropriate encoder
        if self.instruct:
            enc_fn = self.tokenizer.apply_chat_template
        else:
            enc_fn = self.tokenizer
        
        self.model.eval()
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            enc = enc_fn(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)

            input_lengths = enc['attention_mask'].sum(dim=1).cpu().tolist()

            start = time.time()
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            for row, in_len in enumerate(input_lengths):
                out_ids = gen[row, in_len:]
                text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                outputs.append(text)

            print(f"done {len(outputs)} inputs in {time.time() - start: 0.3f}s")
        
        return outputs
