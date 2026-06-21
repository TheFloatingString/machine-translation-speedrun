# Investigating Bilingual Machine Translation Using Small Language Models 

This repository aims to investigate how recent advances in model training can be leveraged to develop lightweight models for machine translation. As part of a Cohere Labs Community Project.

### Ablations 

- Model parameter count 
- Pre-training data (Helsinki-NLP corpus of real-world data, and Aya synthetic data)
- Post-training strategy 
- Swapping attention with SSMs 
- Choice of tokenizer 
- Internal activations such as SwiGLU
- Whether small language models can "remember" enough internal knowledge for more than uni-directional, bilingual translation 
- Residual connections 
- Attention gating 
- Using Muon-like optimizers over Adam

### Empirical Results 
- Diffusion models are much harder to train than autoregressive models at the ~100M scale 
- At the small language model scale, typically encoders are required 
