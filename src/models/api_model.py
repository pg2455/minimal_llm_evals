import logging
import requests
import time
from typing import List, Union
from transformers import AutoTokenizer

from .base import Model


class APIModel(Model):
    def __init__(self, model_name: str, base_url: str):
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            raise ValueError("Invalid base_url: must start with 'http://' or 'https://'")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{base_url}/v1/chat/completions"
        self.completions_url = f"{base_url}/v1/completions"
        self.use_chat_endpoint = True 
        self._test_endpoint() # update chat to completions if chat doesn't work

    def _test_endpoint(self):
        """Test if Chat API is working. If not use completions API."""
        try:
            payload = {
                'model': self.model_name,
                'messages': [
                    {'role': "system", "content": "You are an AI assistant!"},
                    {'role': "user", "content": "This is a test."},
                ],
                "temperature": 0.1,
                "top_p": 0.5,
                "max_tokens":10,
            }
            response = requests.post(self.chat_url, json=payload)
            if response.status_code == 200:
                self.use_chat_template = True
                logging.info("Using chat API...")
                return
            else:
                logging.info(f"Chat endpoint resulted in error: {response.status_code}, Response: {response.text}")
        except:
            pass
        
        self.use_chat_template = False
        logging.info("Using completions API...")

    def predict(
        self, 
        prompts: Union[str, List[str]], 
        batch_size: int = 16, 
        **kwargs
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        outputs: List[str] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            payload = {
                'model': self.model_name,
                'max_tokens': kwargs.get("max_tokens", 1000),
                'temperature': kwargs.get("temperature", 0.7),
                'top_p': kwargs.get("top_p", 0.9),
                'n': kwargs.get("n", 1),
                'stop': kwargs.get("stop", None),
            }
        
            # if its a chat payload, apply chat template here
            if isinstance(batch_prompts[0], list):
                payload['prompt'] = [self.tokenizer.apply_chat_template(x, add_generation_prompt=True, tokenize=False) for x in batch_prompts]
            else:
                payload['prompt'] = batch_prompts

            try:
                start = time.time()
                response = requests.post(self.completions_url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()

                for choice in result['choices']:
                    # since we are using completions api, we clean the text
                    text = self.clean_up_text(choice['text'].strip())
                    # print(text)
                    outputs.append(text)

                logging.info(f"Processing complete. Took {time.time()-start: 0.3f}s. Generated {len(outputs)} so far.")
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed: {e}")
                outputs.extend([""] * len(batch_prompts))

        return outputs

    def clean_up_text(self, text: str):
        cleaned = self.tokenizer.clean_up_tokenization(text)
        special_tokens = self.tokenizer.special_tokens_map.values()
        for token in special_tokens:
            if isinstance(token, str):
                cleaned = cleaned.replace(token, '')
            elif isinstance(token, list):
                for t in token:
                    cleaned = cleaned.replace(t, '')
        
        return cleaned.strip()
