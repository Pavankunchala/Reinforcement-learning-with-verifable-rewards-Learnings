# Quick Start Guide - Financial Reasoning Enhanced

## 🚀 Get Running in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Training (Uses Built-in Data)
```bash
python financial_reasoning_enhanced.py \
    --base-model "unsloth/gemma-3-270m-it" \
    --output-dir "./my_financial_model" \
    --use-4bit
```

### 3. Test Your Model
After training, the script automatically runs sanity checks and shows sample outputs.

## 🎯 What This Does

- **Phase 1**: SFT training on financial reasoning examples
- **Phase 2**: GRPO reinforcement learning with sophisticated rewards
- **Output**: Model that generates structured financial analysis with `<REASONING>`, `<SENTIMENT>`, and `<CONFIDENCE>` tags

## 📊 Sample Output

**Input**: "Tech company reports 25% revenue growth but 8% profit decline due to R&D investment"

**Expected Output**:
```
<REASONING> Revenue growth is strong, reflecting demand and expansion. Profit dipped due to R&D, which is strategic with long-term upside. Near-term margins compress, but growth story remains intact. </REASONING>
<SENTIMENT> positive </SENTIMENT>
<CONFIDENCE> 0.75 </CONFIDENCE>
```

## ⚡ Quick Customization

- **Change model**: `--base-model "your/model"`
- **Adjust training**: `--sft-epochs 5 --grpo-epochs 3.0`
- **Use custom data**: `--train-jsonl "./your_data.jsonl"`

## 🔧 Troubleshooting

- **Memory issues**: Reduce batch sizes or ensure `--use-4bit` is set
- **Slow training**: Start with smaller models or reduce epochs
- **Poor output**: Check that your data follows the expected format

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed configuration options
- Check the [main repository](../README.md) for other RL approaches
- Experiment with different reward function weights and training parameters

---

**Need help?** Check the main README or experiment with the built-in synthetic data first!
