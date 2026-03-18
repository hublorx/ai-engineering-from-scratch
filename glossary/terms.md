# AI Engineering Glossary

## A

### Agent
- **What people say:** "An autonomous AI that thinks and acts on its own"
- **What it actually means:** A while loop where an LLM decides what tool to call next, executes it, sees the result, and repeats
- **Why it's called that:** Borrowed from philosophy — an "agent" is anything that can act in the world. In AI, it just means "LLM + tools + loop"

### Attention
- **What people say:** "How the AI focuses on important parts"
- **What it actually means:** A mechanism where every token computes a weighted sum of all other tokens' values, with weights determined by how relevant they are (via dot product of query and key vectors)
- **Why it's called that:** The 2017 paper "Attention Is All You Need" named it by analogy to human selective attention

### Alignment
- **What people say:** "Making AI safe"
- **What it actually means:** The technical challenge of making an AI system's behavior match human intentions, values, and preferences — including edge cases the designer didn't anticipate

## B

### Backpropagation
- **What people say:** "How neural networks learn"
- **What it actually means:** An algorithm that computes how much each weight contributed to the error by applying the chain rule backward through the network, then adjusts weights proportionally
- **Why it's called that:** Errors propagate backward from output to input, layer by layer

## C

### Context Window
- **What people say:** "How much the AI can remember"
- **What it actually means:** The maximum number of tokens (input + output) that fit in a single API call. Not memory — it's a fixed-size buffer that resets every call

### Chain of Thought (CoT)
- **What people say:** "Making the AI think step by step"
- **What it actually means:** A prompting technique where you ask the model to show its reasoning steps, which improves accuracy on multi-step problems because each step conditions the next token generation

## D

### Diffusion Model
- **What people say:** "AI that generates images from noise"
- **What it actually means:** A model trained to reverse a gradual noising process — it learns to predict and remove noise, and at generation time starts from pure noise and iteratively denoises

### DPO (Direct Preference Optimization)
- **What people say:** "A simpler RLHF"
- **What it actually means:** A training method that skips the reward model entirely — it directly optimizes the language model to prefer the better response in pairs of human preferences

## E

### Embedding
- **What people say:** "Some AI magic that turns words into numbers"
- **What it actually means:** A learned mapping from discrete items (words, images, users) to dense vectors in continuous space, where similar items end up close together
- **Why it's called that:** The items are "embedded" in a geometric space where distance has meaning

## F

### Fine-tuning
- **What people say:** "Training the AI on your data"
- **What it actually means:** Starting with a pre-trained model's weights and continuing training on a smaller, task-specific dataset. Only updates existing weights, doesn't add new knowledge from scratch

## G

### GPT
- **What people say:** "ChatGPT" or "The AI"
- **What it actually means:** Generative Pre-trained Transformer — a specific architecture that predicts the next token using a decoder-only transformer trained on large text corpora
- **Why it's called that:** Generative (produces text), Pre-trained (trained once on large data, then adapted), Transformer (the architecture)

### Gradient Descent
- **What people say:** "How AI improves"
- **What it actually means:** An optimization algorithm that adjusts parameters in the direction that reduces the loss function most steeply, like walking downhill in a high-dimensional landscape

## H

### Hallucination
- **What people say:** "The AI is lying" or "making things up"
- **What it actually means:** The model generates plausible-sounding text that isn't grounded in its training data or the given context — it's pattern-completing, not fact-retrieving

## L

### LLM (Large Language Model)
- **What people say:** "AI" or "the brain"
- **What it actually means:** A transformer-based neural network trained to predict the next token in a sequence, with billions of parameters, trained on internet-scale text data

### LoRA (Low-Rank Adaptation)
- **What people say:** "Efficient fine-tuning"
- **What it actually means:** Instead of updating all weights, insert small low-rank matrices alongside the original weights. Only these small matrices are trained, reducing memory by 10-100x

## M

### MCP (Model Context Protocol)
- **What people say:** "A way for AI to use tools"
- **What it actually means:** An open protocol (JSON-RPC over stdio/HTTP) that standardizes how AI applications connect to external data sources and tools, with typed schemas for tools, resources, and prompts

## P

### Prompt Engineering
- **What people say:** "Talking to AI the right way"
- **What it actually means:** Designing the input text to reliably produce desired outputs — including system prompts, few-shot examples, format instructions, and chain-of-thought triggers

## R

### RAG (Retrieval-Augmented Generation)
- **What people say:** "AI that can search"
- **What it actually means:** A pattern where you retrieve relevant documents from a knowledge base (using embedding similarity), stuff them into the prompt, and let the LLM answer based on that context
- **Why it's called that:** Retrieval (find documents) + Augmented (add to prompt) + Generation (LLM writes the answer)

### RLHF (Reinforcement Learning from Human Feedback)
- **What people say:** "How they make AI helpful"
- **What it actually means:** A training pipeline: (1) collect human preferences on model outputs, (2) train a reward model on those preferences, (3) use PPO to optimize the LLM to produce higher-reward outputs

## S

### Swarm
- **What people say:** "A bunch of AI agents working together like bees"
- **What it actually means:** Multiple agents sharing state and coordinating through message passing, with emergent behavior arising from simple individual rules rather than central control

## T

### Token
- **What people say:** "A word"
- **What it actually means:** A subword unit (typically 3-4 characters in English) produced by a tokenizer like BPE. "unbelievable" might be 3 tokens: "un" + "believ" + "able"

### Transformer
- **What people say:** "The architecture behind modern AI"
- **What it actually means:** A neural network architecture that processes sequences using self-attention (letting every position attend to every other position) instead of recurrence, enabling massive parallelization
- **Why it's called that:** It transforms input representations into output representations through attention layers — and the name sounded cool

## V

### Vector Database
- **What people say:** "A special database for AI"
- **What it actually means:** A database optimized for storing vectors (dense arrays of floats) and performing fast approximate nearest-neighbor search — the core operation in similarity search, RAG, and recommendation systems
