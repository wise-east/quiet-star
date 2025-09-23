# Quiet-STaR

Code for [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629).

This project is implemented by simply patching the base Mistral implementation in Huggingface `transformers` using a new `modeling_mistral.py` and a new `configuration_mistral.py` and otherwise applying standard `transformers` features (e.g. the default Trainer). Our patches were applied to Huggingface's `transformers` version `4.37.0.dev0` under `src/transformers/models/mistral/` -- we cannot guarantee that other changes to their implementation will not affect our implementation, so for reproducibility, we encourage using the same version.

One pitfall to be wary of: the model is not taught not to generate start and end thought tokens. Thus, when performing actual inference, it is necessary to mask these out.

We make an 8-thought-token ahead (including start and end tokens) model [available via Huggingface](https://huggingface.co/ezelikman/quietstar-8-ahead).


## Setup 

```bash 
# on gpu machine 
conda create -n quietstar python=3.11
conda activate quietstar
pip install torch 
pip install -r requirements.txt
huggingface-cli login # for mistral model
```


## Training 

```bash 
# when just using lora with small n_ahead and n_ahead_talk, runs on a single a40 with 48gb memory
python quiet-star-train.py --max_length 256 --lora --learning_rate 1e-4 --lora_r 64 --lora_alpha 128 --n_ahead 4 --n_ahead_talk 2

# when using lora with largeer n_ahead and n_ahead_talk, runs on a single a100 with 80gb memory
python quiet-star-train.py --max_length 128 --lora --learning_rate 1e-4 --lora_r 64 --lora_alpha 128 --n_ahead 8 --n_ahead_talk 4
```


## Testing base 
```bash 
python quiet-star-train.py --original --max_length 256 
```

This should return ~6% accuracy on GSM8K and ~36% accuracy on CSQA, which is in line with the results in the paper.
```
Trial evaluation metrics: {'eval_gsm8k_loss': 4.394712924957275, 'eval_gsm8k_accuracy': 0.06264498829841614, 'eval_gsm8k_runtime': 77.9772, 'eval_gsm8k_samples_per_second': 2.565, 'eval_gsm8k_steps_per_second': 2.565, 'eval_csqa_loss': 5.664874076843262, 'eval_csqa_accuracy': 0.36201953887939453, 'eval_csqa_runtime': 14.1808, 'eval_csqa_samples_per_second': 14.104, 'eval_csqa_steps_per_second': 14.104}
```
