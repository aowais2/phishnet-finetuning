# Tiny LLM fine tuned with RL

We consider small language models (LMs) for finetuning with Reinforcement Learning (RL) methods. Small language models in this context refer to 3B-8B paramater models.

Small LMs do not perform well on phishing detection tasks out of the box [1]. Phishing detection using smaller LMs has been proposed before [1], however these methods only utilize simpler methods without finetuning, such as prompt engineering.

To remedy this, finetuning methods such as Low Rank Adaptation (LORA) are proposed. This, along with minimal further finetuning with RL methods using Direct Preference Optimization (DPO) [2] will be utilized. This is in contrast to other time and data expensive methods such as RLHF (RL from Human Feedback).

Other methods were considered, such as Retrieval Augmented Generation (RAG). However, these were deemed insufficient because the problem statement is defined as a classification problem (phishing vs regular email classification), not a knowledge retrieval issue.

Further, we propose the utilization of DPO as opposed to other RL methods such as PPO or GRPO because GRPO is mainly used for complex reasoning improvements, and adds unnecessary complexity for a binary classification task.

The following pipeline is proposed for finetuning a small LM for phishing detection:

1.   Supervised finetuning with LoRA on phishing vs legitimate labels.
2.   Direct Preference Optimization (DPO)
3.   Evaluation of the model

[1] Lin, Zijie, Zikang Liu, and Hanbo Fan. "Improving Phishing Email Detection Performance of Small Large Language Models." arXiv preprint arXiv:2505.00034 (2025).

[2] Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model." Advances in neural information processing systems 36 (2023): 53728-53741.
