"""
Inference utilities for testing trained models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, List


class InferenceEngine:
    """Simple inference engine for testing post-trained models."""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the trained model (or LoRA adapter)
            base_model_path: If using LoRA, path to base model
            device: Device to use ('auto', 'cuda', 'cpu')
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
        """
        self.device = device

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model from {model_path}...")

        # Quantization config
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        # Load model
        if base_model_path:
            # LoRA adapter
            print(f"Loading base model from {base_model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs
            )
            print(f"Loading LoRA adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
        else:
            # Full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )

        self.model.eval()
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to sample (False = greedy)

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def chat(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate response in instruction-following format.

        Args:
            instruction: The instruction/question
            input_text: Optional input context
            max_new_tokens: Max tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Model's response
        """
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def compare_responses(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        **kwargs
    ):
        """
        Generate and compare responses to multiple prompts.

        Args:
            prompts: List of prompts to test
            max_new_tokens: Max tokens per generation
            **kwargs: Additional generation parameters
        """
        print("\n" + "="*80)
        print("COMPARING RESPONSES")
        print("="*80)

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[Prompt {i}]")
            print(f"{prompt}")
            print(f"\n[Response {i}]")
            response = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            print(response)
            print("-"*80)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Test inference on trained models")
    parser.add_argument("--model", required=True, help="Path to model or LoRA adapter")
    parser.add_argument("--base-model", help="Path to base model (if using LoRA)")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    # Initialize engine
    engine = InferenceEngine(
        model_path=args.model,
        base_model_path=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type 'quit' to exit\n")

        while True:
            instruction = input("Instruction: ").strip()
            if instruction.lower() == 'quit':
                break

            input_text = input("Input (optional): ").strip()

            response = engine.chat(
                instruction,
                input_text,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

            print(f"\nResponse:\n{response}\n")
            print("-"*80)

    elif args.prompt:
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\nPrompt:\n{args.prompt}")
        print(f"\nResponse:\n{response}\n")

    else:
        # Default: test with sample prompts
        test_prompts = [
            "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
            "### Instruction:\nWrite a Python function to calculate fibonacci numbers.\n\n### Response:\n",
            "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
        ]

        engine.compare_responses(test_prompts, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
