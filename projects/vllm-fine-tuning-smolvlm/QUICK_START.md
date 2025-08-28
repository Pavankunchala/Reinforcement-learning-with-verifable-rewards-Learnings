# Quick Start: SmolVLM-256M ChartQA Fine-Tuning

## 🚀 Get Training in 5 Minutes

### 1. Navigate to the Project
```bash
cd Reinforcement-learning-with-verifable-rewards-Learnings/projects/vllm-fine-tuning-smolvlm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Training (Optimized for 16GB GPU)
```bash
python train_smolvlm_chartqa.py
```

**That's it!** The script will:
- ✅ Download SmolVLM-256M model
- ✅ Load ChartQA dataset with lazy loading
- ✅ Fine-tune with optimized parameters
- ✅ Save model to `./smolvlm-256m-chartqa-sft/`

## 📊 Expected Output

```
🚀 High-Performance Training Configuration:
   • Batch Size: 16
   • Learning Rate: 0.001
   • Epochs: 2
   • LoRA Rank (r): 16
   • Memory Target: 14.0GB

🚀 Starting HIGH-PERFORMANCE SmolVLM-256M training...
✅ Training completed successfully!
💾 Saving final model to smolvlm-256m-chartqa-sft
```

## 🧪 Test Your Model

### Quick Test (10 samples)
```bash
python test_smolvlm_chartqa.py
```

### Extended Test (50 samples)
```bash
python test_smolvlm_chartqa.py --num_samples 50
```

## ⚡ Performance Tips

### For Different GPU Sizes

#### 16GB GPU (Recommended)
```bash
python train_smolvlm_chartqa.py --memory_limit 14.0 --batch_size 16
```

#### 12GB GPU
```bash
python train_smolvlm_chartqa.py --memory_limit 10.0 --batch_size 12
```

#### 8GB GPU
```bash
python train_smolvlm_chartqa.py --memory_limit 6.0 --batch_size 8
```

### For Better Accuracy

#### High-Capacity LoRA
```bash
python train_smolvlm_chartqa.py --lora_r 24 --lora_alpha 48 --epochs 3
```

#### Longer Training
```bash
python train_smolvlm_chartqa.py --max_steps 750 --learning_rate 0.001
```

## 🎯 What You Get

After training, you'll have:
- **Fine-tuned SmolVLM-256M** model in `./smolvlm-256m-chartqa-sft/`
- **Test results** in `./smolvlm_256m_test_output/`
- **Performance metrics** and sample predictions
- **Ready-to-use model** for chart understanding tasks

## 📈 Expected Performance

| Metric | Expected | Actual (Recent Test) |
|--------|----------|---------------------|
| Training Time | 15-25 min | ~12 min |
| GPU Memory | <2GB | ~0.5GB |
| Test Accuracy | 40-60% | 40% (4/10) |
| Model Size | 256M params | 256M params |

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Memory Errors
```bash
# Reduce batch size and memory limit
python train_smolvlm_chartqa.py --batch_size 8 --memory_limit 6.0
```

#### Slow Training
```bash
# Use higher learning rate and larger batch size
python train_smolvlm_chartqa.py --learning_rate 0.002 --batch_size 20
```

#### Poor Performance
```bash
# Increase LoRA capacity and training time
python train_smolvlm_chartqa.py --lora_r 24 --lora_alpha 48 --epochs 3 --max_steps 750
```

## 📚 Next Steps

1. **Read the full [README.md](README.md)** for detailed configuration options
2. **Experiment with different LoRA settings** for your specific use case
3. **Check the [main repository](../../README.md)** for other RL approaches
4. **Share your results** and contribute improvements!

## 🎉 Success Metrics

**Your training is successful if:**
- ✅ Model loads without errors
- ✅ Training completes in 15-25 minutes
- ✅ GPU memory usage stays under your specified limit
- ✅ Test accuracy is above 30%
- ✅ Model generates reasonable chart-related answers

---

**Happy fine-tuning! 🎯**

*Need help? Check the full README.md or open an issue in the main repository.*
