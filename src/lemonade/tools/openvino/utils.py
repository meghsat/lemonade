import os
import time
import json
import logging
from queue import Queue
from packaging.version import Version
import openvino_genai as ov_genai
from transformers import AutoTokenizer
from lemonade.tools.adapter import (
    ModelAdapter,
    TokenizerAdapter,
    PassthroughTokenizerResult,
)


class OpenVinoTokenizer(TokenizerAdapter):
    def __init__(self, model_path, hf_tokenizer: AutoTokenizer):
        super().__init__(hf_tokenizer)
        # Initialize OpenVino tokenizer
        self.tokenizer = ov_genai.Tokenizer(model_path)

        # Placeholder value since some code will try to query it
        # If we actually need this to return a proper value, then
        # ov_genai.GeneratorParams.eos_token_id has it
        self.eos_token_id = None

    def __call__(self, prompt: str, return_tensors="np"):
        tokens = self.tokenizer.encode(prompt)
        return PassthroughTokenizerResult(tokens)

    # pylint: disable=unused-argument
    def decode(self, response, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(response)


class OpenVinoStreamer:
    def __init__(self, tokenizer: OpenVinoTokenizer, timeout=None):
        self.tokenizer = tokenizer
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def add_text(self, text: str):
        self.text_queue.put(text, timeout=self.timeout)

    def done(self):
        self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class OpenVinoModel(ModelAdapter):

    def __init__(self, input_folder):
        super().__init__()
        self.pipe = self.load_pipeline(input_folder, "CPU", None, None)
        self.type = "ov-genai"
        self.config = self.load_config(input_folder)

    def __init__(self, input_folder, device, max_prompt_len, min_response_len):
        super().__init__()
        self.pipe = self.load_pipeline(
            input_folder, device, max_prompt_len, min_response_len
        )
        self.type = "ov-genai"
        self.config = self.load_config(input_folder)

    def load_pipeline(
        self, input_folder, device, max_prompt_len=2048, min_response_len=128
    ):
        # build Intel's LLM Pipeline
        # max input length of the pipeline must be determined in "max_prompt_len" field in pipe config
        print("Creating LLMPipeline:")
        print(f"LLMPipeline Max Input Prompt Len: {max_prompt_len}")

        # require some padding to prevent crashes when passing in the exact prompt
        padding = 10

        # create the pipe configuration used when creating the LLM Pipe
        if device in ["NPU"]:
            pipe_config = {
                "GENERATE_HINT": "BEST_PERF",
                "MAX_PROMPT_LEN": int(max_prompt_len) + padding,
                "MIN_RESPONSE_LEN": int(min_response_len),
            }
        else:
            pipe_config = {
                "KV_CACHE_PRECISION": "u8"
            }
            

        pipe = ov_genai.LLMPipeline(input_folder, device, **pipe_config)
        return pipe

    def load_config(self, input_folder):
        max_prompt_length = 4096

        config_path = os.path.join(input_folder, "generation_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                config_dict["max_prompt"] = max_prompt_length
                return config_dict
        return None

    def generate(
        self,
        input_ids,
        max_new_tokens=512,
        min_new_tokens=0,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.7,
        streamer: OpenVinoStreamer = None,
        pad_token_id=None,
        stopping_criteria=None,
        max_length=None,
        random_seed=1,
    ):
        pipe = self.pipe
        generation_config = ov_genai.GenerationConfig()

        generation_config.max_new_tokens = max_new_tokens
        generation_config.min_new_tokens = min_new_tokens
        generation_config.do_sample = do_sample
        generation_config.temperature = temperature

        if streamer is None:
            # execute generation on the llm pipe with given generation configurations
            start_time = time.time()
            result = pipe.generate(input_ids, generation_config)
            end_time = time.time()
            return [result, start_time, end_time]
        else:
            # TODO: if streamer is present do...
            raise NotImplementedError
