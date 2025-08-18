# Financial Reasoning Enhanced: SFT + GRPO Fine-Tuning

This project demonstrates advanced fine-tuning of language models for financial reasoning tasks using a two-phase approach: Supervised Fine-Tuning (SFT) followed by GRPO (Group Relative Policy Optimization) with multi-level reward functions.

## Overview

The Financial Reasoning Enhanced pipeline combines:
- **SFT Phase**: Initial training on structured financial reasoning examples
- **GRPO Phase**: Reinforcement learning with sophisticated reward functions including:
  - Format compliance gates
  - Financial reasoning quality analysis
  - FinBERT teacher alignment
  - Confidence calibration
  - Directional consistency validation

## Key Features

- **Multi-level Reward System**: Combines format, reasoning quality, and financial expertise
- **FinBERT Integration**: Uses FinBERT as a teacher model for sentiment alignment
- **Structured Output**: Enforces consistent `<REASONING>`, `<SENTIMENT>`, and `<CONFIDENCE>` tags
- **Unsloth Integration**: Optimized for 4-bit quantization and efficient training
- **Flexible Data Sources**: Supports Financial PhraseBank, synthetic data, or custom JSONL

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional Dependencies**:
   - `seaborn` and `matplotlib` for visualization (will work without them)
   - `bitsandbytes` for 4-bit quantization (optional, Unsloth handles this)

## Usage

### Basic Training Pipeline

```bash
python financial_reasoning_enhanced.py \
    --base-model "unsloth/gemma-3-270m-it" \
    --output-dir "./financial_reasoning_outputs" \
    --use-4bit \
    --data-mode "mixed"
```

### Advanced Configuration

```bash
python financial_reasoning_enhanced.py \
    --base-model "unsloth/gemma-3-270m-it" \
    --output-dir "./custom_outputs" \
    --use-4bit \
    --sft-epochs 5 \
    --grpo-epochs 3.0 \
    --sft-batch 8 \
    --grpo-batch 4 \
    --sft-lr 2e-4 \
    --grpo-lr 1e-5 \
    --beta 0.2 \
    --temperature 0.8 \
    --data-mode "real" \
    --max-real-examples 500
```

### Custom Data Training

```bash
python financial_reasoning_enhanced.py \
    --base-model "unsloth/gemma-3-270m-it" \
    --output-dir "./custom_training" \
    --use-4bit \
    --train-jsonl "./custom_sft_data.jsonl" \
    --eval-jsonl "./custom_grpo_data.jsonl"
```

## Data Format

### SFT Training Data (JSONL)
Each line should contain:
```json
{
    "text": "Tech company reports 25% revenue growth but 8% profit decline due to R&D investment",
    "reasoning": "Revenue growth is strong, reflecting demand and expansion. Profit dipped due to R&D, which is strategic with long-term upside.",
    "sentiment": "positive",
    "confidence": 0.75
}
```

### GRPO Evaluation Data (JSONL)
Each line should contain:
```json
{
    "text": "Bank announces 3% dividend increase while facing regulatory scrutiny over compliance issues"
}
```

## Output Structure

The model learns to generate responses in this exact format:
```
<REASONING> Revenue growth is strong but margins compressed due to higher input costs; guidance is cautious, suggesting near-term volatility. Therefore, outlook is balanced with upside from new products. </REASONING>
<SENTIMENT> neutral </SENTIMENT>
<CONFIDENCE> 0.72 </CONFIDENCE>
```

## Key Arguments

### Model & Training
- `--base-model`: Base model identifier (default: unsloth/gemma-3-270m-it)
- `--output-dir`: Output directory for checkpoints and final model
- `--use-4bit`: Enable 4-bit quantization for memory efficiency
- `--local-only`: Use only local files (offline mode)

### SFT Parameters
- `--sft-epochs`: Number of SFT training epochs (default: 3)
- `--sft-batch`: SFT batch size (default: 12)
- `--sft-lr`: SFT learning rate (default: 1e-4)
- `--sft-warmup`: SFT warmup ratio (default: 0.1)

### GRPO Parameters
- `--grpo-epochs`: Number of GRPO training epochs (default: 4.0)
- `--grpo-batch`: GRPO batch size (default: 12)
- `--grpo-lr`: GRPO learning rate (default: 1e-5)
- `--beta`: KL penalty coefficient (default: 0.15)
- `--temperature`: Generation temperature (default: 0.7)
- `--num-generations`: Completions per prompt (default: 6)

### Data Control
- `--data-mode`: Data source selection ["mixed", "real", "synthetic"]
- `--max-real-examples`: Maximum examples from Financial PhraseBank (default: 200)
- `--train-jsonl`: Custom SFT training data file
- `--eval-jsonl`: Custom GRPO evaluation data file

### LoRA Configuration
- `--lora-rank`: LoRA rank for parameter-efficient training (default: 32)
- `--lora-alpha`: LoRA alpha scaling factor (default: 64)

## Reward Function Components

1. **Format Gate (35%)**: Ensures proper tag structure
2. **Financial Reasoning (25%)**: Quality, logic, and context analysis
3. **Sentiment Alignment (20%)**: FinBERT teacher agreement
4. **Confidence Calibration (15%)**: Brier score-like confidence accuracy
5. **Directional Consistency (5%)**: Reasoning-sentiment alignment

## Example Outputs

### Input
```
"Energy company reports 30% production increase but faces environmental lawsuit"
```

### Expected Output
```
<REASONING> Production increase signals operational efficiency and market demand growth, which is positive for revenue. However, environmental lawsuit introduces regulatory risk and potential financial liabilities that could offset gains. </REASONING>
<SENTIMENT> neutral </SENTIMENT>
<CONFIDENCE> 0.68 </CONFIDENCE>
```

## Performance Monitoring

The training pipeline provides:
- Real-time loss and reward tracking
- KL divergence monitoring
- Memory usage statistics
- Training progress visualization (if matplotlib/seaborn available)

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch sizes or enable 4-bit quantization
2. **Training Instability**: Lower learning rates or increase beta for KL penalty
3. **Poor Format Compliance**: Increase format gate weight in reward function
4. **Slow Convergence**: Adjust warmup ratios or learning rate schedules

### Performance Tips
- Use 4-bit quantization for memory efficiency
- Start with smaller models for experimentation
- Use synthetic data for initial testing
- Monitor reward components to identify training issues

## Integration with Other Projects

This project complements the TRL PPO fine-tuning example by providing:
- Specialized financial reasoning capabilities
- Multi-level reward system demonstration
- FinBERT integration example
- Structured output enforcement

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{financial_reasoning_enhanced_2025,
  title={Financial Reasoning Enhanced: SFT + GRPO Fine-Tuning},
  author={Pavan Kunchala},
  year={2025},
  url={https://github.com/yourusername/Reinforcement-learning-with-verifable-rewards-Learnings}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
