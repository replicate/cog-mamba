import os
from threading import Thread

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers.generation import TextIteratorStreamer

from cog import BasePredictor, Input, ConcatenateIterator
from utils import download_and_extract


MODEL_NAME = "state-spaces/mamba-130m"
MODEL_DTYPE = torch.float16
MODEL_CACHE = "/src/mamba-model-cache"

MODEL_URL_MAP = {
    "state-spaces/mamba-130m": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-130m.tar",
    "state-spaces/mamba-370m": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-370m.tar",
    "state-spaces/mamba-790m": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-790m.tar",
    "state-spaces/mamba-1.4b": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-1.4b.tar",
    "state-spaces/mamba-2.8b": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-2.8b.tar",
    "state-spaces/mamba-2.8b-slimpj": "https://weights.replicate.delivery/default/mamba/state-spaces-mamba-2.8b-slimpj.tar",
}
TOKENIZER_URL = "https://weights.replicate.delivery/default/mamba/eleutherai-gpt-neoex-20b-tokenizer.tar"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download the model and tokenizer if they are not already cached
        os.makedirs(MODEL_CACHE, exist_ok=True)
        model_path = os.path.join(MODEL_CACHE, ("-").join(MODEL_NAME.split("/")))
        tokenizer_path = os.path.join(MODEL_CACHE, "EleutherAI-gpt-neox-20b")

        if not os.path.exists(model_path):
            checkpoint_url = MODEL_URL_MAP[MODEL_NAME]
            download_and_extract(checkpoint_url, model_path)

        if not os.path.exists(tokenizer_path):
            print(f"Downloading tokenizer from {TOKENIZER_URL}")
            download_and_extract(TOKENIZER_URL, tokenizer_path)

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
        )
        self.model = MambaLMHeadModel.from_pretrained(
            model_path, device=self.device, dtype=MODEL_DTYPE
        )
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Text prompt to send to the model."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens.",
            ge=1,
            le=5000,
            default=100,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.1,
            le=5.0,
            default=1.0,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.",
            default=1,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=10.0,
            default=1.2,
        ),
        seed: int = Input(
            description="The seed for the random number generator", default=None
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        if seed == None:  # use a random seed
            seed = torch.randint(0, 100000, (1,)).item()
        torch.random.manual_seed(seed)

        tokens = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        input_ids = tokens.input_ids.to(device=self.device)
        max_length = input_ids.shape[1] + max_length

        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for _, new_text in enumerate(streamer):
            yield new_text

        thread.join()
