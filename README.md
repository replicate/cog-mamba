# Cog wrapper for Mamba LLMs
This is a cog wrapper for Mamba LLM models. See the original [repo](https://github.com/state-spaces/mamba), [paper](https://arxiv.org/abs/2312.00752) and Replicate [demo](https://replicate.com/adirik/mamba-130m) for details.


## Basic Usage
You will need to have [Cog](https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md) and Docker installed to serve your model as an API. Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of the model to [Replicate](https://replicate.com) with Cog. To run a prediction:

```bash
cog predict -i prompt="How are you doing today?"
```

To start your server and serve the model as an API:
```bash
cog run -p 5000 python -m cog.server.http
```

The API input arguments are as follows:

- **prompt:** The text prompt for Mamba.  
- **max_length:** Maximum number of tokens to generate. A word is generally 2-3 tokens.  
- **temperature:** Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.  
- **top_p:** Samples from the top p percentage of most likely tokens during text decoding, lower to ignore less likely tokens.  
- **top_k:** Samples from the top k most likely tokens during text decoding, lower to ignore less likely tokens.  
- **repetition_penalty:** Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.   
- **seed:** The seed parameter for deterministic text generation. A specific seed can be used to reproduce results or left blank for random generation.  


## References
```
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```