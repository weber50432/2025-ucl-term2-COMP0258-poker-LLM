import os
import torch
import re
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# ----------------------- Configurations -----------------------
max_seq_length = 1024   # Can increase for longer reasoning traces
lora_rank = 64          # Larger rank = smarter, but slower

# Total training steps (for demonstration, you can set this to a higher number)
TOTAL_TRAINING_STEPS = 250
# Evaluate after every 10% of the total training steps.
EVAL_INTERVAL = TOTAL_TRAINING_STEPS // 10  

# Evaluation dataset sizes (set to None to use full dataset)
TRAIN_SUBSET_SIZE = 32 
TEST_SUBSET_SIZE = 32

# ----------------------- Data Preparation -----------------------
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def get_pokerbench_questions(split="train") -> Dataset:
    # Load PokerBench dataset from Hugging Face
    data = load_dataset('RZ412/PokerBench', split=split)
    # Format dataset: each sample now has a 'prompt' and an 'answer'
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['instruction']}
        ],
        'answer': x['output'].strip()
    })
    return data

# ----------------------- Reward Functions -----------------------
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    # Lowercase and strip both extracted answer and ground truth
    extracted_responses = [extract_xml_answer(r).lower().strip() for r in responses]
    ground_truths = [a.lower().strip() for a in answer]
    print('-'*20, f"Question:\n{q}",
          f"\nGround Truth Answer:\n{ground_truths[0]}",
          f"\nGenerated Response:\n{responses[0]}",
          f"\nExtracted Answer:\n{extracted_responses[0]}")

    rewards = []
    for er, gt in zip(extracted_responses, ground_truths):
        # Full match: if the ground truth is found within the extracted answer.
        if gt in er:
            rewards.append(2.0)
        else:
            # Extract the action token (first word) from both.
            er_tokens = er.split()
            gt_tokens = gt.split()
            if er_tokens and gt_tokens:
                er_action = er_tokens[0]
                gt_action = gt_tokens[0]
                # If the action words match and are valid poker actions, return half reward.
                if er_action == gt_action and er_action in ["bet", "raise", "check", "call", "fold"]:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
    return rewards

def poker_action_validator(completions, **kwargs) -> list[float]:
    """Validate poker action format (e.g., 'bet 18', 'check', 'fold')."""
    pattern = r"^(bet|raise|check|call|fold)( \d+)?$"
    responses = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    matches = [re.fullmatch(pattern, r.lower()) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ----------------------- Model and Trainer Setup -----------------------
def create_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,           # False for LoRA 16bit
        fast_inference=True,         # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,  # Adjust if needed
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )
    return model, tokenizer

def create_trainer(model, tokenizer, train_dataset, steps):
    # Create a GRPO configuration with the given number of training steps
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=400,
        num_train_epochs=1,
        max_steps=steps,  # Train for a given number of steps in this call
        save_steps=steps,  # Save at the end of these steps
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
            poker_action_validator,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )
    return trainer

# ----------------------- Evaluation Function -----------------------
def evaluate_model(model, dataset, tokenizer, use_lora=False, checkpoint_path=None):
    """
    If checkpoint_path is provided, load the LoRA checkpoint once;
    otherwise, if use_lora is True, load the default "grpo_saved_lora" per sample.
    """
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    if checkpoint_path is not None:
        # Load the checkpoint once
        lora_request = model.load_lora(checkpoint_path)
    else:
        lora_request = model.load_lora("grpo_saved_lora") if use_lora else None

    correct_count = 0
    total = 0
    for sample in dataset:
        prompt = sample['prompt']
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        reward = correctness_reward_func([prompt], [[{"content": output}]], [sample['answer']])[0]
        if reward == 2.0:
            correct_count += 1
        total += 1
    accuracy = correct_count / total if total > 0 else 0.0
    return accuracy

# ----------------------- Main Script -----------------------
if __name__ == "__main__":
    mode = input("Enter 'train' to train a new model or 'eval' to evaluate a saved model: ").strip().lower()
    
    if mode == "eval":
        # ----------------------- Evaluation Mode -----------------------
        print("Running in evaluation mode.")
        # Prepare evaluation datasets (optionally subset for quicker evaluation)
        train_eval_dataset = get_pokerbench_questions(split="train")
        test_eval_dataset = get_pokerbench_questions(split="test")
        if TRAIN_SUBSET_SIZE is not None:
            train_eval_dataset = train_eval_dataset.select(range(TRAIN_SUBSET_SIZE))
            print(f"Using a subset of {TRAIN_SUBSET_SIZE} training examples for evaluation.")
        if TEST_SUBSET_SIZE is not None:
            test_eval_dataset = test_eval_dataset.select(range(TEST_SUBSET_SIZE))
            print(f"Using a subset of {TEST_SUBSET_SIZE} test examples for evaluation.")

        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer()
        
        checkpoint_path = input("Enter the checkpoint path for the saved model (default 'grpo_saved_lora'): ").strip()
        if not checkpoint_path:
            checkpoint_path = "grpo_saved_lora"
        
        train_acc = evaluate_model(model, train_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
        test_acc = evaluate_model(model, test_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
        print(f"Evaluation results:\n  Training Accuracy = {train_acc:.2%}\n  Testing Accuracy = {test_acc:.2%}")
        exit(0)
    
    elif mode == "train":
        # ----------------------- Training Mode -----------------------
        # Load the full training dataset (will be used for training)
        full_train_dataset = get_pokerbench_questions(split="train")
        # Prepare evaluation datasets (optionally subset for quicker evaluation)
        train_eval_dataset = get_pokerbench_questions(split="train")
        test_eval_dataset = get_pokerbench_questions(split="test")
        if TRAIN_SUBSET_SIZE is not None:
            train_eval_dataset = train_eval_dataset.select(range(TRAIN_SUBSET_SIZE))
            print(f"Using a subset of {TRAIN_SUBSET_SIZE} training examples for evaluation.")
        if TEST_SUBSET_SIZE is not None:
            test_eval_dataset = test_eval_dataset.select(range(TEST_SUBSET_SIZE))
            print(f"Using a subset of {TEST_SUBSET_SIZE} test examples for evaluation.")

        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer()

        # We'll record (training_step, train_accuracy, test_accuracy)
        accuracy_log = []
        current_step = 0

        # Evaluate after the first training step (base performance)
        print("Performing evaluation after the first training step (base performance)...")
        trainer = create_trainer(model, tokenizer, full_train_dataset, steps=1)
        trainer.train()  # Run 1 training step
        current_step += 1
        # Save checkpoint after step 1
        checkpoint_path = f"grpo_checkpoint_{current_step}"
        model.save_lora(checkpoint_path)
        base_train_acc = evaluate_model(model, train_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
        base_test_acc = evaluate_model(model, test_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
        accuracy_log.append((current_step, base_train_acc, base_test_acc))
        print(f"Step {current_step}: Training Acc = {base_train_acc:.2%}, Testing Acc = {base_test_acc:.2%}")

        # Continue training in increments (each increment equals 10% of TOTAL_TRAINING_STEPS)
        while current_step < TOTAL_TRAINING_STEPS:
            steps_to_train = EVAL_INTERVAL  # e.g. 10 steps per evaluation interval
            trainer = create_trainer(model, tokenizer, full_train_dataset, steps=steps_to_train)
            trainer.train()  # Train for the next increment
            current_step += steps_to_train
            checkpoint_path = f"grpo_checkpoint_{current_step}"
            model.save_lora(checkpoint_path)
            train_acc = evaluate_model(model, train_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
            test_acc = evaluate_model(model, test_eval_dataset, tokenizer, checkpoint_path=checkpoint_path)
            accuracy_log.append((current_step, train_acc, test_acc))
            print(f"Step {current_step}: Training Acc = {train_acc:.2%}, Testing Acc = {test_acc:.2%}")

        # ----------------------- Plotting the Results -----------------------
        steps = [entry[0] for entry in accuracy_log]
        train_accuracies = [entry[1] for entry in accuracy_log]
        test_accuracies = [entry[2] for entry in accuracy_log]

        plt.figure(figsize=(8, 6))
        plt.plot(steps, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(steps, test_accuracies, marker='o', label='Test Accuracy')
        plt.xlabel("Training Steps")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy vs. Training Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig("accuracy_vs_steps.png")
        plt.show()
    
    else:
        print("Invalid mode. Please enter either 'train' or 'eval'.")
        exit(1)
