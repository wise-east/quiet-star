import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os
import time
import wandb
from transformers import EarlyStoppingCallback
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
from argparse import ArgumentParser

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

from transformers.models.mistral import configuration_mistral as original_configuration_mistral
from transformers.models.mistral import modeling_mistral as original_modeling_mistral


parser = ArgumentParser()
parser.add_argument("--n_ahead", type=int, default=8)
parser.add_argument("--n_ahead_talk", type=int, default=4)
parser.add_argument("--n_passes", type=int, default=2)
parser.add_argument("--n_examples", type=int, default=1_000)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--full_batch_size", type=int, default=8)
parser.add_argument("--eval_and_logging_steps", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--original", action="store_true", help="Use original Mistral model instead of Quiet-STaR")
parser.add_argument("--lora", action="store_true", help="Use LoRA")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--save_steps", type=int, default=10000)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training via DeepSpeed")
args = parser.parse_args()

# MAIN SETUP
root_prefix = "./"
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
n_ahead_talk_global = args.n_ahead_talk
n_passes_global = args.n_passes
n_ahead_global = args.n_ahead
n_examples = args.n_examples
per_device_train_batch_size = args.per_device_train_batch_size
full_batch_size = args.full_batch_size
eval_and_logging_steps = args.eval_and_logging_steps
save_steps = args.save_steps

def model_init(params):
    original = False
    if params is None:
        params = {}
    else: 
        # unpack params as dict 
        params = vars(params)

    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "mistralai/Mistral-7B-v0.1"
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map='auto',
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        attn_implementation="sdpa",
    )
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.kill_after = 1000
    model.train()
    # Apply LoRA (PEFT) to reduce trainable parameters and memory footprint
    if params.get("lora", False):
        lora_config = LoraConfig(
            r=params.get("lora_r", 8),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        for n, p in model.named_parameters():
            if n in ["start_embedding", "end_embedding"] or n.startswith("talk_head"):
                p.requires_grad = True
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model

# Load dataset
dataset = load_dataset(
    dataset_name,
    "en" if "c4" in dataset_name else "default",
    split=f"train[:{n_examples}]",
    num_proc=16
)

max_length = args.max_length
train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200, fn_kwargs={"max_length": max_length})
eval_dataset_gsm = load_dataset("gsm8k", "main", split="test[:200]").map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200, fn_kwargs={"max_length": max_length})
eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split="validation[:200]").map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200, fn_kwargs={"max_length": max_length})

eval_datasets = {
    "gsm8k": eval_dataset_gsm,
    "csqa": eval_dataset_csqa,
}

global_gradient_accumulation_steps = full_batch_size // per_device_train_batch_size
run_id = int(time.time())

training_args = TrainingArguments(
    output_dir=root_prefix + f"cache/quietstar/{run_id}",
    learning_rate=args.learning_rate,
    # optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    # optim="paged_adamw_8bit",
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=global_gradient_accumulation_steps,
    max_grad_norm=1.0,
    max_steps=1000,
    warmup_steps=20,
    auto_find_batch_size=False,
    weight_decay=0.001,
    label_names=["labels"],
    include_inputs_for_metrics=True,
    logging_steps=1,
    eval_steps=eval_and_logging_steps,
    evaluation_strategy="steps",
    save_steps=save_steps,
    run_name=f"n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}",
    # gradient_checkpointing=True,
    # deepspeed="deepspeed_zero3.json",
    bf16=True,
)

if args.original:
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

else: 
    import configuration_mistral
    import modeling_mistral
    original_modeling_mistral.MistralModel = modeling_mistral.MistralModel
    original_modeling_mistral.MistralForCausalLM = modeling_mistral.MistralForCausalLM
    original_configuration_mistral.MistralConfig = configuration_mistral.MistralConfig

    model = model_init(args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    compute_metrics=compute_metrics,
)

# do a trial evaluation before training to catch issues early
try:
    model.eval()
    print("Running trial evaluation before training...")
    trial_metrics = trainer.evaluate()
    print("Trial evaluation metrics:", trial_metrics)
    model.train()
except Exception as e:
    print("Trial evaluation failed:", e)
    raise

trainer.train()
