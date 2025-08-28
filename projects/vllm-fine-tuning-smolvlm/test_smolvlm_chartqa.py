#!/usr/bin/env python3
"""
🚀 SmolVLM-256M ChartQA Fine-tuning Test Script

This script tests the fine-tuned SmolVLM-256M model from train_smolvlm_chartqa.py
on the ChartQA dataset to evaluate performance and accuracy.

Features:
- Loads the fine-tuned adapter from smolvlm-256m-chartqa-sft
- Tests on ChartQA validation set
- Generates detailed responses with images
- Saves results to JSON and summary files
- Configurable via command-line arguments
"""

import json
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer
from transformers import Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO
import os

# Config - defaults match train_smolvlm_chartqa.py
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"  # Base model from finetuning script
ADAPTER_DIR = "./smolvlm-256m-chartqa-sft"  # LoRA adapter directory from finetuning
SAMPLE_SPLIT = "val[:10%]"  # Use validation split, get 10% for more samples
OUTPUT_DIR = "./smolvlm_256m_test_output"  # Directory to save images and results
NUM_SAMPLES = 10  # Number of samples to process

def load_image(src):
    """Load image from URL, file path, or PIL object."""
    if isinstance(src, str) and src.startswith("http"):
        try:
            resp = requests.get(src, timeout=5)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
    elif isinstance(src, str) and os.path.exists(src):
        try:
            return Image.open(src).convert("RGB")
        except Exception:
            return None
    return None

def main():
    import argparse  # Add argparse for command-line arguments

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test fine-tuned SmolVLM-256M model with configurable parameters.')
    parser.add_argument('--model_id', type=str, default=BASE_MODEL_ID, help=f'Base model ID to use (default: {BASE_MODEL_ID})')
    parser.add_argument('--adapter_dir', type=str, default=ADAPTER_DIR, help=f'LoRA adapter directory (default: {ADAPTER_DIR})')
    parser.add_argument('--memory_limit', type=float, default=2.0, help='Maximum memory limit in GB to monitor and warn (default: 2.0 GB)')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help=f'Number of samples to process (default: {NUM_SAMPLES})')
    args = parser.parse_args()

    # Update configs from arguments
    model_id = args.model_id
    adapter_dir = args.adapter_dir
    memory_limit = args.memory_limit
    num_samples = args.num_samples

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Add memory limit check
    if torch.cuda.is_available() and memory_limit > 0:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb: {int(memory_limit * 1024)}'  # Convert GB to MB for allocation hint

    # Try to load a multimodal processor first, fall back if needed
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        # Tokenizer is often available alongside the processor for decoding
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        use_processor = True
        print("✅ Successfully loaded multimodal processor and tokenizer")
    except Exception as e:
        print(f"⚠️  AutoProcessor load failed: {e}")
        print("   Attempting to use AutoTokenizer only (text-only).")
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        use_processor = False

    # Load the base model
    print(f"Loading base model: {model_id}")
    model = Idefics3ForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)
    print("✅ Base model loaded successfully")

    # Load and apply LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_dir}")
    if os.path.exists(adapter_dir):
        try:
            # Check adapter config first
            adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print("📄 Found adapter config, loading adapter...")
                model = PeftModel.from_pretrained(model, adapter_dir)

                # For inference, merge the adapter weights to avoid issues
                print("🔄 Merging LoRA adapter weights for inference...")
                model = model.merge_and_unload()
                print("✅ LoRA adapter merged successfully")

                # Test if adapter was applied
                total_params = sum(p.numel() for p in model.parameters())
                print(f"📊 Model parameters: {total_params:,} total")
            else:
                print("❌ No adapter_config.json found in adapter directory")

        except Exception as e:
            print(f"❌ Error loading LoRA adapter: {e}")
            print("   Using base model without fine-tuning")
    else:
        print(f"⚠️  LoRA adapter directory not found: {adapter_dir}")
        print("   Using base model without fine-tuning")

    # Load validation subset from ChartQA
    print(f"Loading ChartQA dataset (split: {SAMPLE_SPLIT})...")
    ds = load_dataset("HuggingFaceM4/ChartQA", split=SAMPLE_SPLIT)
    print(f"✅ Dataset loaded: {len(ds)} samples available")

    results = []
    processed_count = 0

    print(f"\n🚀 Starting to process {num_samples} samples...")
    print("=" * 60)

    for idx, sample in enumerate(ds):
        if processed_count >= num_samples:
            break

        image_src = sample.get("image")
        prompt = sample.get("query", "")
        ground_truth = sample.get("label", "")

        print(f"\n📊 Processing Sample {processed_count + 1}/{num_samples} (Dataset index: {idx})")
        print("-" * 50)

        # Load and save image locally
        # Check if image_src is already a PIL Image object (from HuggingFace datasets)
        if hasattr(image_src, 'mode') and hasattr(image_src, 'size'):
            # It's already a PIL Image object
            image = image_src
            print("✅ Image is already loaded as PIL Image object")
        else:
            # Try to load as URL or file path
            image = load_image(image_src)
            if image is None:
                print(f"❌ Could not load image: {image_src}")
                continue

        # Save image locally
        image_filename = f"sample_{processed_count:02d}.png"
        image_path = os.path.join(images_dir, image_filename)
        try:
            image.save(image_path)
            print(f"✅ Image saved: {image_path}")
        except Exception as e:
            print(f"⚠️  Could not save image: {e}")
            image_path = image_src  # Use original URL if save fails

        # Process with model
        try:
            if use_processor:
                # Try a simpler approach for SmolVLM
                # Use the processor directly with text and images
                try:
                    inputs = processor(text=prompt, images=image, return_tensors="pt")
                except Exception as e:
                    print(f"⚠️  Direct processor failed: {e}")
                    print("   Falling back to chat template approach...")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            else:
                # If only text is available, tokenize prompt and run a text-only path
                inputs = tokenizer(prompt, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Debug: show input shapes for first few samples
            if processed_count < 2:
                print(f"🔧 Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,  # Shorter for chart QA answers
                    temperature=0.1,    # Lower temperature for more focused answers
                    do_sample=True,     # Enable sampling but with low temperature
                    top_p=0.9,          # Nucleus sampling
                    top_k=50,           # Top-k sampling
                    repetition_penalty=1.1,  # Reduce repetition
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else tokenizer.pad_token_id
                )

            # Decode and extract just the answer part
            if tokenizer is not None:
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Debug: print raw response for first few samples
                if processed_count < 2:
                    print(f"🔍 Raw model output: '{full_response}'")

                # Extract just the assistant's answer (remove the conversation format)
                if "Assistant:" in full_response:
                    answer = full_response.split("Assistant:")[-1].strip()
                elif "\nAssistant:" in full_response:
                    answer = full_response.split("\nAssistant:")[-1].strip()
                else:
                    # Fallback: try to find the last meaningful answer
                    lines = full_response.strip().split('\n')
                    answer = lines[-1].strip() if lines else full_response.strip()

                # Clean up any remaining artifacts
                answer = answer.replace('<|end_of_text|>', '').replace('<|im_end|>', '').strip()

                # If answer is too long or contains too many tokens, take first reasonable part
                if len(answer.split()) > 10:  # If more than 10 words, likely not a clean answer
                    words = answer.split()
                    # Look for common chart answer patterns (numbers, colors, short phrases)
                    if any(word.isdigit() for word in words[:5]):
                        answer = ' '.join(words[:5])
                    else:
                        answer = words[0] if words else "Unable to extract answer"

            else:
                # Fallback: raw generation output
                answer = str(outputs[0])

            print(f"❓ Question: {prompt}")
            print(f"🎯 Ground Truth: {ground_truth}")
            print(f"🤖 Model Answer: {answer}")
            print(f"🖼️  Image: {image_path}")

            results.append({
                "sample_id": processed_count,
                "dataset_index": idx,
                "image_path": image_path,
                "original_image_url": str(image_src) if not isinstance(image_src, str) else image_src,
                "question": prompt,
                "ground_truth": ground_truth,
                "model_answer": answer,
                "processing_status": "success"
            })

            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            results.append({
                "sample_id": processed_count,
                "dataset_index": idx,
                "image_path": image_path,
                "original_image_url": str(image_src) if not isinstance(image_src, str) else image_src,
                "question": prompt,
                "ground_truth": ground_truth,
                "model_answer": f"ERROR: {str(e)}",
                "processing_status": "error"
            })
            processed_count += 1

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create a clean summary file
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SMOLVLM-256M FINETUNED MODEL TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Adapter: {adapter_dir}\n")
        f.write(f"Total samples processed: {len(results)}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n\n")

        for result in results:
            f.write(f"SAMPLE {result['sample_id'] + 1:02d}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Image: {result['image_path']}\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Model Answer: {result['model_answer']}\n")
            f.write(f"Status: {result['processing_status']}\n\n")

    print("\n" + "=" * 60)
    print("🎉 TESTING COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📊 Processed {len(results)} samples")
    print(f"🖼️  Images saved to: {images_dir}")
    print(f"📋 Detailed results: {results_path}")
    print(f"📝 Summary: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
