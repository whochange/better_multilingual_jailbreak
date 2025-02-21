from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.base_model import BaseModel
from src.utils.gpu_utils import clear_gpu_memory
from typing import Optional, Union

class TransformerGenerator(BaseModel):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_path = model_config['model_path']
        self.max_length = model_config.get('max_length', 128)
        self.temperature = model_config.get('temperature', 0)
        self.batch_size = model_config.get('batch_size', 8)
        self.chat_template = model_config.get('chat_template', 'default')
        self.system_prompt = model_config.get('system_prompt', None)
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer based on model type."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if "aya-101" in self.model_path:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
            
            self.to_device()
        except Exception as e:
            raise RuntimeError(f"Error loading model {self.model_path}: {e}")
        
    def _format_chat_input(self, text: str, system_prompt: Optional[str] = None) -> Union[str, List[Dict]]:
        """Format input based on model's chat template."""
        if self.chat_template == "aya-expanse":
            return [{"role": "user", "content": text}]
        
        elif self.chat_template == "seallm":
            if system_prompt:
                return f"<|im_start|>system\n{system_prompt}</s><|im_start|>user\n{text}</s><|im_start|>assistant\n"
            return f"<|im_start|>user\n{text}</s><|im_start|>assistant\n"
            
        elif self.chat_template == "llama-2":
            if system_prompt:
                return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{text} [/INST]"
            return f"[INST] {text} [/INST]"
        
        elif self.chat_template == "sw-llama-2":
            if system_prompt:
                return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{text} [/INST]"
            return f"[INST] {text} [/INST]"
            
        elif self.chat_template == "llama-3":
            if system_prompt:
                return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"          
        
        elif self.chat_template == "qwen":
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
        else:  # default
            return f"{system_prompt}\n\n{text}" if system_prompt else text
        
    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        """Generate outputs for the given inputs."""
        outputs = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            if isinstance(self.model, AutoModelForSeq2SeqLM) or self.chat_template == "aya-101":
                # T5 model handling
               #print(f"Generating with AutoModelForSeq2SeqLM")
                encoded = self.tokenizer.batch_encode_plus(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        **kwargs
                    )
            elif self.chat_template == "sw-llama-2":
                # use _format_chat_input to format the input one by one
                formatted_batch = []
                for text in batch:
                    formatted = self._format_chat_input(text, self.system_prompt)
                    formatted_batch.append(formatted)
                
                encoded = self.tokenizer(
                    formatted_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        **kwargs
                    )
            elif self.chat_template == "llama-2":
                # Special handling for llama-2
                formatted_batch = []
                for text in batch:
                    formatted = self._format_chat_input(text, self.system_prompt)
                    formatted_batch.append(formatted)
                
                encoded = self.tokenizer(
                    formatted_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
            else:
                # Causal LM handling
                #print(f"Generating with AutoModelForCausalLM")
                formatted_batch = []
                for text in batch:
                    formatted = [{'role': 'user', 'content': text}]#self._format_chat_input(text, self.system_prompt)
                    if isinstance(formatted, list):
                        # Apply chat template for message dicts
                        formatted = self.tokenizer.apply_chat_template(
                            formatted,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    formatted_batch.append(formatted)
                
                encoded = self.tokenizer(
                    formatted_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
            
            # Decode and clean responses
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for text, response in zip(batch, decoded):
                # Handle different chat templates to extract the assistant's response
                response = response.replace(text, "").strip()
                if self.chat_template in ['llama-3', 'seallm', 'qwen']:
                    if "assistant" in response:
                        response = response.split("assistant", 1)[1].strip()
                elif self.chat_template in ["sw-llama-2"]:
                    response = response.split("[/INST]", 1)[1].strip()
                elif self.chat_template == "aya-expanse":
                    response = response.split("<|CHATBOT_TOKEN|>")[1].strip()
                    # Remove input text if present (fallback)
                    
                
                outputs.append(response)
            # input_lengths = encoded['attention_mask'].sum(dim=1).tolist()
            # for i, (text, output_ids_seq) in enumerate(zip(batch, output_ids)):
            #     response_ids = output_ids_seq[input_lengths[i]:]
            #     response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            #     outputs.append(response)
            
            # clear_gpu_memory() if i + self.batch_size < len(inputs) else None
        
        return outputs
        
if __name__ == "__main__":
    # Test the generator directly
    from config.model_config import DEFAULT_MODELS
    
    print("Testing TransformerGenerator...")
    config = DEFAULT_MODELS['aya'].__dict__
    
    generator = TransformerGenerator(config)
    test_inputs = ["Translate to English: Bonjour le monde", "How are you?"]
    
    outputs = generator.generate(test_inputs)
    print("\nTest inputs:", test_inputs)
    print("Generated outputs:", outputs)