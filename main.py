import modal
from dataclasses import dataclass

app = modal.App("transformer-speedrun")
volume = modal.Volume.from_name("transformer-speedrun-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("transformers[torch]")
    .uv_pip_install("datasets")
    .uv_pip_install("wandb")
    .uv_pip_install("pyyaml")
    .uv_pip_install("sentencepiece")
    .uv_pip_install("evaluate")
    .uv_pip_install("sacrebleu")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("cfg", remote_path="/root/cfg")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train(model_type: str = "gpt2", force_tokenize: bool = False, muon: bool = True):
    from datasets import load_dataset, load_from_disk
    from transformers import (
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import os
    import yaml
    import wandb
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import math

    # Import custom model from src
    from src.model import GPT2Modded, GPT2Config as GPT2ConfigModded

    # 1. Setup tokenizer first to check cache
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load or tokenize the dataset
    cache_path = "/data/tokenized_fineweb_1b"
    
    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        if force_tokenize:
            print("Force tokenize flag set. Re-tokenizing dataset...")
        print("Loading FineWeb (sample-10BT) dataset...")
        # Load 10% of the 10BT sample to get ~1B tokens
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
        
        print("Subsetting to 5% of sample-10BT (~500M tokens)...")
        # Using 5% to approximate 500M tokens (5% of 10B)
        dataset = dataset.select(range(int(len(dataset) * 0.05)))
        
        # FineWeb only has a 'train' split, so we create our own validation set
        print("Creating validation split...")
        split_ds = dataset.train_test_split(test_size=0.005, seed=42)
        from datasets import DatasetDict
        ds = DatasetDict({
            "train": split_ds["train"],
            "validation": split_ds["test"]
        })

        # 3. Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        print("Tokenizing dataset...")
        tokenized_datasets = ds.map(
            tokenize_function, batched=True, num_proc=8, remove_columns=["text"]
        )

        # Filter out empty examples
        tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)
        
        print(f"Saving tokenized dataset to volume: {cache_path}")
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit() # Ensure data is written to the volume

    # 4. Initialize the model
    # Configuration matches the requested parameters
    config_params = {
        "vocab_size": tokenizer.vocab_size,
        "n_positions": 512,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
    }

    # Check for YAML config in cfg directory
    config_from_yaml = False
    run_name = f"transformer-speedrun-{model_type}"
    # Check if model_type is a path to a yaml file or a key in cfg
    potential_yaml_paths = [
        os.path.join("/root/cfg", f"{model_type}.yaml"),
        os.path.join("/root/cfg", model_type if model_type.endswith(".yaml") else f"{model_type}.yaml"),
        model_type if model_type.endswith(".yaml") else None
    ]
    
    for yaml_path in potential_yaml_paths:
        if yaml_path and os.path.exists(yaml_path):
            print(f"Loading model configuration from {yaml_path}...")
            # Use the filename as the run name if we load from YAML
            run_name = os.path.basename(yaml_path).replace(".yaml", "")
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                if "name" in yaml_config:
                    run_name = yaml_config["name"]
                if "hyperparams" in yaml_config:
                    config_params.update(yaml_config["hyperparams"])
                else:
                    config_params.update(yaml_config)
            config_from_yaml = True
            break

    # Use the appropriate Config class
    if model_type == "gpt2_modded" or config_from_yaml:
        print(f"Using custom GPT2Modded architecture with params: {config_params}")
        config = GPT2ConfigModded(**config_params)
        model = GPT2Modded(config)
    else:
        print(f"Using standard HuggingFace GPT2LMHeadModel with params: {config_params}")
        # HF GPT2Config uses slightly different attribute names for some things, 
        # but the ones we use (n_embd, n_layer, n_head, n_positions) are standard.
        config = GPT2Config(**config_params)
        model = GPT2LMHeadModel(config)

    # In multi-GPU setups, the Trainer handles device placement.
    # We just log the availability here.
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} GPUs.")
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")
    # Store model size in config so it's logged to wandb by the Trainer
    model.config.model_params = model_size

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,            # 1 epoch of 100M tokens is ~200k steps at batch 512, but we'll limit by max_steps if needed
        per_device_train_batch_size=8, # Increased for A100-80GB
        gradient_accumulation_steps=8,  # 64 * 8 = 512 effective batch size
        gradient_checkpointing=True,   # Huge memory saver
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=5e-4,            # This will be overridden by custom optimizer if we pass it
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,              # Reduced warmup for shorter run
        report_to="wandb",
        run_name=run_name,
        eval_strategy="steps",
        eval_steps=100,
        max_steps=20000000,                # Limit steps for the speedrun (20M)
    )

    # 7. Custom Optimizer Setup (Muon + AdamW)
    from src.optimizer import Muon, CombinedOptimizer
    
    def get_optimizers(model, learning_rate, weight_decay, use_muon=True):
        if not use_muon:
            print("Muon disabled. Using AdamW for all parameters.")
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Filter parameters for Muon (only 2D parameters in transformer layers)
        muon_params = []
        adamw_params = []
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Muon handles 2D parameters (weights of Linear layers)
            # but we usually exclude embeddings and the head
            if "transformer.h" in name and len(p.shape) == 2 and "ln" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        
        optimizer_muon = Muon(muon_params, lr=0.02, momentum=0.95)
        optimizer_adamw = torch.optim.AdamW(adamw_params, lr=learning_rate, weight_decay=weight_decay)
        
        return CombinedOptimizer([optimizer_muon, optimizer_adamw])

    optimizer = get_optimizers(model, training_args.learning_rate, training_args.weight_decay, use_muon=muon)

    # 8. Initialize Trainer
    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 50 == 0:
                print(f"\n--- Step {state.global_step} ---")
                print("Generating sample sentences...")
                model = kwargs["model"]

                model.eval()
                prompts = [
                    "The quick brown fox",
                    "Artificial intelligence is",
                    "The history of the world",
                ]

                for prompt in prompts:
                    # Get device from model parameters (works for both HF models and custom nn.Modules)
                    device = next(model.parameters()).device
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=50,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7,
                        )
                    generated_text = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}")
                model.train()  # Switch back to training mode
                print("-" * 30)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        # HF Trainer multiplies loss by gradient_accumulation_steps in
                        # training_step but does not divide it back before logging, so
                        # we correct for that here. This callback is inserted at index 0
                        # so it runs before WandbCallback and the corrected value is
                        # what gets sent to wandb.
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                        logs["train_perplexity"] = math.exp(logs["loss"])
                    if "eval_loss" in logs:
                        logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, math.OverflowError):
                    pass

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        optimizers=(optimizer, None), # Trainer accepts (optimizer, scheduler)
        callbacks=[GenerationCallback(tokenizer)],
    )
    # Insert before WandbCallback so logs["loss"] is corrected before wandb reads it
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    # 8. Train
    print("Starting pre-training...")
    trainer.train()

    # 9. Save the model
    trainer.save_model("./final_model")
    print("Training complete! Model saved to ./final_model")


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train_mt(force_tokenize: bool = False, muon: bool = True, model_cfg: str = ""):
    from datasets import load_dataset, load_from_disk
    from transformers import (
        MarianTokenizer,
        BartConfig,
        BartForConditionalGeneration,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    import evaluate as hf_evaluate
    import os
    import yaml
    import torch
    import math
    import numpy as np

    from src.optimizer import Muon, CombinedOptimizer

    # 1. Tokenizer (~60K vocab, handles EN + FR)
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    # 2. Load or tokenize dataset
    # New cache path — format differs from the old decoder-only approach
    cache_path = "/data/tokenized_opus100_en_fr_seq2seq"

    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        if force_tokenize:
            print("Force tokenize flag set. Re-tokenizing dataset...")
        print("Loading opus-100 en-fr dataset...")
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

        def tokenize_fn(examples):
            srcs = [t["en"] for t in examples["translation"]]
            tgts = [t["fr"] for t in examples["translation"]]
            model_inputs = tokenizer(srcs, max_length=128, truncation=True)
            label_encodings = tokenizer(tgts, max_length=128, truncation=True)
            model_inputs["labels"] = label_encodings["input_ids"]
            return model_inputs

        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(
            tokenize_fn, batched=True, num_proc=8, remove_columns=["translation"]
        )

        print(f"Saving tokenized dataset to volume: {cache_path}")
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit()

    # 3. Model config — BART encoder-decoder; defaults to a small model
    #    YAML hyperparams (n_embd / n_layer / n_head) are mapped to BART names
    config_params = {
        "d_model": 512,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "decoder_ffn_dim": 2048,
        "max_position_embeddings": 256,
    }
    run_name = "mt-en-fr-bart-small"

    if model_cfg:
        yaml_paths = [
            os.path.join("/root/cfg", f"{model_cfg}.yaml"),
            os.path.join("/root/cfg", model_cfg if model_cfg.endswith(".yaml") else f"{model_cfg}.yaml"),
            model_cfg if model_cfg.endswith(".yaml") else None,
        ]
        for yaml_path in yaml_paths:
            if yaml_path and os.path.exists(yaml_path):
                print(f"Loading model config from {yaml_path}...")
                run_name = os.path.basename(yaml_path).replace(".yaml", "") + "-mt"
                with open(yaml_path) as f:
                    yc = yaml.safe_load(f)
                    if "name" in yc:
                        run_name = yc["name"] + "-mt"
                    params = yc.get("hyperparams", yc)
                    # Map GPT-2-style names → BART names
                    if "n_embd" in params:
                        d = params.pop("n_embd")
                        config_params["d_model"] = d
                        config_params["encoder_ffn_dim"] = 4 * d
                        config_params["decoder_ffn_dim"] = 4 * d
                    if "n_layer" in params:
                        n = params.pop("n_layer")
                        config_params["encoder_layers"] = n
                        config_params["decoder_layers"] = n
                    if "n_head" in params:
                        h = params.pop("n_head")
                        config_params["encoder_attention_heads"] = h
                        config_params["decoder_attention_heads"] = h
                    if "n_positions" in params:
                        config_params["max_position_embeddings"] = params.pop("n_positions")
                    params.pop("vocab_size", None)
                    config_params.update(params)
                break

    # Use eos_token_id (=0, within SP vocab range) for pad and decoder start.
    # MarianTokenizer.pad_token_id == vocab_size (one past the end of the SP model),
    # so using it in BartConfig causes "piece id is out of range" when the trainer
    # tries to decode padded prediction arrays.
    eos_id = tokenizer.eos_token_id
    bart_config = BartConfig(
        vocab_size=tokenizer.vocab_size,
        decoder_start_token_id=eos_id,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        forced_eos_token_id=eos_id,
        **config_params,
    )
    model = BartForConditionalGeneration(bart_config)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")

    # 4. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # 5. BLEU metric (evaluated via generate during eval steps)
    sacrebleu = hf_evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Clip to valid SP vocab range — the trainer may pad with pad_token_id
        # which can equal vocab_size and be out of range for the SP model.
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        return {"bleu": round(result["score"], 2)}

    # 6. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_mt",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        report_to="wandb",
        run_name=run_name,
        eval_strategy="steps",
        eval_steps=500,
        predict_with_generate=True,
        generation_max_length=128,
    )

    # 7. Optimizer — Muon for 2D attention/FFN weights, AdamW for embeddings/norms.
    #    Track parameter ids to avoid double-counting tied weights (embedding ↔ lm_head).
    def get_optimizers(model, lr, wd, use_muon=True):
        if not use_muon:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        seen_ids = set()
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if len(p.shape) == 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        return CombinedOptimizer([
            Muon(muon_params, lr=0.02, momentum=0.95),
            torch.optim.AdamW(adamw_params, lr=lr, weight_decay=wd),
        ])

    optimizer = get_optimizers(model, training_args.learning_rate, training_args.weight_decay, use_muon=False)

    # 8. Callbacks
    class MTGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 50 == 0:
                model = kwargs["model"]
                model.eval()
                device = next(model.parameters()).device

                examples = [
                    "Hello, how are you?",
                    "The weather is beautiful today.",
                    "I would like to order a coffee, please.",
                ]
                print(f"\n--- MT Samples @ Step {state.global_step} ---")
                for src in examples:
                    enc = self.tokenizer(src, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(
                            **enc,
                            max_new_tokens=64,
                            num_beams=4,
                            early_stopping=True,
                        )
                    translation = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"EN: {src}")
                    print(f"FR: {translation}\n")
                model.train()
                print("-" * 40)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                        logs["train_perplexity"] = math.exp(logs["loss"])
                    if "eval_loss" in logs:
                        logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, math.OverflowError):
                    pass

    # 9. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[MTGenerationCallback(tokenizer)],
    )
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    # 10. Train
    print("Starting EN→FR seq2seq machine translation training (AdamW)...")
    trainer.train()

    trainer.save_model("./final_mt_model")
    print("MT training complete! Model saved to ./final_mt_model")


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train_diffusion(force_tokenize: bool = False):
    from datasets import load_from_disk, load_dataset
    from transformers import (
        MarianTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import os
    import torch
    import math
    from torch.nn.utils.rnn import pad_sequence

    from src.diffusion import TextDiffusionMT, DiffusionConfig

    # 1. Tokenizer (same as MT — EN+FR, ~60K vocab)
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    eos_id = tokenizer.eos_token_id  # = 0, safely within SP vocab range

    # 2. Dataset — reuse seq2seq cache (same input_ids / labels format)
    cache_path = "/data/tokenized_opus100_en_fr_seq2seq"

    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        print("Tokenizing opus-100 en-fr dataset...")
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

        def tokenize_fn(examples):
            srcs = [t["en"] for t in examples["translation"]]
            tgts = [t["fr"] for t in examples["translation"]]
            model_inputs = tokenizer(srcs, max_length=128, truncation=True)
            label_encodings = tokenizer(tgts, max_length=128, truncation=True)
            model_inputs["labels"] = label_encodings["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["translation"])
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit()

    # 3. Model
    config = DiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_enc_layers=6,
        n_den_layers=6,
        n_heads=8,
        ffn_dim=2048,
        max_len=128,
        T=2000,
    )
    model = TextDiffusionMT(config)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")

    # 4. Data collator — pad input_ids with eos_id (in-vocab), labels with -100
    def diff_collate_fn(batch):
        input_ids = pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=eos_id)
        attention_mask = torch.zeros_like(input_ids)
        for i, x in enumerate(batch):
            attention_mask[i, :len(x["input_ids"])] = 1
        labels = pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./results_diffusion",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        report_to="wandb",
        run_name="diffusion-mt-en-fr",
        eval_strategy="steps",
        eval_steps=500,
    )

    # 6. Callbacks
    class DiffusionGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 50 == 0:
                model = kwargs["model"]
                model.eval()
                device = next(model.parameters()).device

                examples = [
                    "Hello, how are you?",
                    "The weather is beautiful today.",
                    "I would like to order a coffee, please.",
                ]
                print(f"\n--- Diffusion MT Samples @ Step {state.global_step} (50 DDIM steps) ---")
                for src in examples:
                    enc = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        token_ids = model.ddim_sample(enc["input_ids"], enc["attention_mask"], tgt_len=32, steps=50)
                    ids = token_ids[0].tolist()
                    translation = self.tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"EN: {src}")
                    print(f"FR: {translation!r} (ids: {ids[:10]}...)")
                model.train()
                print("-" * 40)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                    if "eval_loss" in logs:
                        logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, math.OverflowError):
                    pass

    # 7. Trainer — subclass to avoid safetensors error on tied embedding/lm_head weights
    class DiffusionTrainer(Trainer):
        def _save(self, output_dir, state_dict=None):
            import json
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.model.config.to_dict(), f, indent=2)

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        data_collator=diff_collate_fn,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        callbacks=[DiffusionGenerationCallback(tokenizer)],
    )
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    print("Starting DDIM text diffusion MT training...")
    trainer.train()

    trainer.save_model("./final_diffusion_model")
    print("Diffusion MT training complete! Model saved to ./final_diffusion_model")


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train_mdlm(force_tokenize: bool = False):
    from datasets import load_from_disk, load_dataset
    from transformers import (
        MarianTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import os
    import torch
    import math
    from torch.nn.utils.rnn import pad_sequence

    from src.mdlm import MaskedDiffusionMT, MDLMConfig

    # 1. Tokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    eos_id = tokenizer.eos_token_id  # = 0

    # 2. Dataset — reuse seq2seq cache
    cache_path = "/data/tokenized_opus100_en_fr_seq2seq"

    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        print("Tokenizing opus-100 en-fr dataset...")
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

        def tokenize_fn(examples):
            srcs = [t["en"] for t in examples["translation"]]
            tgts = [t["fr"] for t in examples["translation"]]
            model_inputs = tokenizer(srcs, max_length=128, truncation=True)
            label_encodings = tokenizer(tgts, max_length=128, truncation=True)
            model_inputs["labels"] = label_encodings["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["translation"])
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit()

    # 3. Model
    config = MDLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_enc_layers=6,
        n_den_layers=6,
        n_heads=8,
        ffn_dim=2048,
        max_len=128,
        T=1000,
    )
    model = MaskedDiffusionMT(config)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")

    # 4. Data collator — same as diffusion: pad input_ids with eos_id, labels with -100
    def mdlm_collate_fn(batch):
        input_ids = pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=eos_id)
        attention_mask = torch.zeros_like(input_ids)
        for i, x in enumerate(batch):
            attention_mask[i, :len(x["input_ids"])] = 1
        labels = pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./results_mdlm",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        report_to="wandb",
        run_name="mdlm-mt-en-fr",
        eval_strategy="steps",
        eval_steps=500,
    )

    # 6. Callbacks
    class MDLMGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 50 == 0:
                model = kwargs["model"]
                model.eval()
                device = next(model.parameters()).device

                examples = [
                    "Hello, how are you?",
                    "The weather is beautiful today.",
                    "I would like to order a coffee, please.",
                ]
                print(f"\n--- MDLM MT Samples @ Step {state.global_step} (50 unmasking steps) ---")
                eos = self.tokenizer.eos_token_id  # = 0
                for src in examples:
                    enc = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        token_ids = model.sample(enc["input_ids"], enc["attention_mask"], tgt_len=32, steps=50, repetition_penalty=1.3)
                    raw_ids = token_ids[0].tolist()
                    # Truncate at EOS only if it appears after position 0 — at
                    # early training the model collapses to all-EOS and we still
                    # want to see the raw ids rather than an empty string.
                    eos_pos = next((i for i, x in enumerate(raw_ids) if x == eos and i > 0), None)
                    ids = raw_ids[:eos_pos] if eos_pos is not None else raw_ids
                    translation = self.tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"EN: {src}")
                    print(f"FR: {translation!r} (raw ids: {raw_ids[:10]}...)")
                model.train()
                print("-" * 40)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                except (OverflowError, math.OverflowError):
                    pass

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=mdlm_collate_fn,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        callbacks=[MDLMGenerationCallback(tokenizer)],
    )
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    print("Starting masked discrete diffusion (MDLM) MT training...")
    trainer.train()

    trainer.save_model("./final_mdlm_model")
    print("MDLM training complete! Model saved to ./final_mdlm_model")


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train_diffuseq(force_tokenize: bool = False):
    from datasets import load_from_disk, load_dataset
    from transformers import (
        MarianTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import os
    import json
    import torch
    import math
    from torch.nn.utils.rnn import pad_sequence

    from src.diffuseq import DiffuSeqMT, DiffuSeqConfig

    # 1. Tokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    eos_id = tokenizer.eos_token_id

    # 2. Dataset — reuse seq2seq cache
    cache_path = "/data/tokenized_opus100_en_fr_seq2seq"
    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        print("Tokenizing opus-100 en-fr dataset...")
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

        def tokenize_fn(examples):
            srcs = [t["en"] for t in examples["translation"]]
            tgts = [t["fr"] for t in examples["translation"]]
            model_inputs = tokenizer(srcs, max_length=128, truncation=True)
            label_encodings = tokenizer(tgts, max_length=128, truncation=True)
            model_inputs["labels"] = label_encodings["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["translation"])
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit()

    # 3. Model
    config = DiffuSeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_layers=12,
        n_heads=8,
        ffn_dim=2048,
        max_src_len=128,
        max_tgt_len=128,
        T=2000,
    )
    model = DiffuSeqMT(config)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")

    # 4. Data collator
    def diffuseq_collate_fn(batch):
        input_ids = pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=eos_id)
        attention_mask = torch.zeros_like(input_ids)
        for i, x in enumerate(batch):
            attention_mask[i, :len(x["input_ids"])] = 1
        labels = pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./results_diffuseq",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        report_to="wandb",
        run_name="diffuseq-mt-en-fr",
        eval_strategy="steps",
        eval_steps=500,
    )

    # 6. Callbacks
    class DiffuSeqGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 50 == 0:
                model = kwargs["model"]
                model.eval()
                device = next(model.parameters()).device

                examples = [
                    "Hello, how are you?",
                    "The weather is beautiful today.",
                    "I would like to order a coffee, please.",
                ]
                print(f"\n--- DiffuSeq MT Samples @ Step {state.global_step} (50 DDIM steps) ---")
                for src in examples:
                    enc = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        token_ids = model.ddim_sample(enc["input_ids"], enc["attention_mask"], tgt_len=32, steps=50)
                    ids = token_ids[0].tolist()
                    translation = self.tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"EN: {src}")
                    print(f"FR: {translation!r} (ids: {ids[:10]}...)")
                model.train()
                print("-" * 40)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                        logs["train_perplexity"] = math.exp(logs["loss"])
                    if "eval_loss" in logs:
                        logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, math.OverflowError):
                    pass

    # 7. Trainer — torch.save to avoid safetensors tied-weight error
    class DiffuSeqTrainer(Trainer):
        def _save(self, output_dir, state_dict=None):
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.model.config.to_dict(), f, indent=2)

    trainer = DiffuSeqTrainer(
        model=model,
        args=training_args,
        data_collator=diffuseq_collate_fn,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        callbacks=[DiffuSeqGenerationCallback(tokenizer)],
    )
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    print("Starting DiffuSeq MT training...")
    trainer.train()

    trainer.save_model("./final_diffuseq_model")
    print("DiffuSeq training complete! Model saved to ./final_diffuseq_model")


@app.local_entrypoint()
def main(model: str = "gpt2", force_tokenize: bool = False, muon: bool = True, task: str = "lm", model_cfg: str = ""):
    if task == "mt":
        train_mt.remote(force_tokenize=force_tokenize, muon=muon, model_cfg=model_cfg)
    elif task == "diff":
        train_diffusion.remote(force_tokenize=force_tokenize)
    elif task == "mdlm":
        train_mdlm.remote(force_tokenize=force_tokenize)
    elif task == "diffuseq":
        train_diffuseq.remote(force_tokenize=force_tokenize)
    else:
        train.remote(model_type=model, force_tokenize=force_tokenize, muon=muon)
