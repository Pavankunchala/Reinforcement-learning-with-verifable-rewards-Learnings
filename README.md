# Reinforcement Learning Learnings

This repository is a collection of my personal projects and experiments in the field of Reinforcement Learning (RL), particularly focusing on techniques involving language models and fine-tuning approaches, including vision-language models and memory-efficient training strategies.

## Projects

The projects are organized into subdirectories within the `projects/` folder. Each project is self-contained with its own README and dependency list.

-   **[TRL PPO Fine-Tuning](./projects/trl-ppo-fine-tuning/)**: An example of fine-tuning a language model using Proximal Policy Optimization (PPO) with the `trl` library.
-   **[Financial Reasoning Enhanced](./projects/financial-reasoning-enhanced/)**: Advanced fine-tuning pipeline combining SFT and GRPO with multi-level reward functions for financial reasoning tasks.
-   **[vLLM Fine-Tuning SmolVLM](./projects/vllm-fine-tuning-smolvlm/)**: Ultra-efficient fine-tuning of SmolVLM-256M on ChartQA using lazy loading and streaming for maximum memory efficiency on consumer GPUs.

## Approaches Covered

- **PPO (Proximal Policy Optimization)**: Traditional reinforcement learning for language model fine-tuning
- **GRPO (Group Relative Policy Optimization)**: Advanced RLHF technique with group-based sampling
- **SFT + RL Hybrid**: Supervised fine-tuning followed by reinforcement learning for optimal performance
- **Vision-Language Fine-Tuning**: Efficient fine-tuning of multimodal models for chart understanding and visual reasoning tasks
- **Memory-Efficient Training**: Lazy loading and streaming techniques for training large models on limited hardware
- **Multi-level Rewards**: Sophisticated reward functions combining format compliance, reasoning quality, and domain expertise

---

*This repository is actively being updated.*