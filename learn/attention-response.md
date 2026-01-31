# Attention in Transformer - Elaborated Answers

**Generated:** 2026-01-31 15:00 UTC

**Original Resource:** https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7

---

## Your Current Understanding

- Embedding could represent meaning
- Attention(Q,K,V): Q is input query, K is key/word in the model, V is value for the weight
- Masking before softmax is needed to avoid the sum is not 1
- The size of the parameters is the square of the context size
- Calculate the value and use it as delta to improve the model

---

## Questions & Elaborated Answers

### 1. Low rank transformation - why the part of the matrix could be dropped?

Low-rank transformations in the transformer attention mechanism are effective because parts of the attention matrix can be approximated or compressed without significantly impacting performance. This is primarily due to the observation that the attention scores often lie in a low-dimensional eigenspace, meaning that most of the variation among attention scores can be captured by a low-rank representation.

This low-rank structure implies that many entries in the attention matrix are redundant or can be approximated by a smaller set of basis vectors, allowing parts of the matrix to be dropped or compressed. Research shows that the attention matrices tend to have a low-rank factorization or can be efficiently approximated using low-rank methods, such as low-rank factorization or low-rank key/value representations.

These methods leverage the fact that the principal components of the attention scores contain most of the relevant information, enabling the removal of less significant components or parts of the matrix while maintaining the model's performance. This approach reduces computational costs and memory requirements, making it feasible to scale transformers to larger models without a proportional increase in complexity.

In essence, the low-rank assumption exploits the inherent structure of attention matrices, where most of the meaningful information is concentrated in a small subspace, allowing parts of the matrix to be dropped or approximated efficiently while preserving the model's effectiveness.

**Learn more:**
- [Low-Rank Bottleneck in Multi-head Attention Models](https://proceedings.mlr.press/v119/bhojanapalli20a/bhojanapalli20a.pdf)
- [arXiv:1912.00835](https://arxiv.org/abs/1912.00835)

---

### 2. Cross attention - is it important compared to self-attention? Can't understand the difference

The key difference between self-attention and cross-attention lies in their roles and how they process information:

**Self-attention** allows a sequence to interpret itself by weighing the importance of each element relative to others within the same sequence, enabling the model to understand internal relationships. This is what you see in models like BERT and GPT - the model learns how different words in a sentence relate to each other.

**Cross-attention**, in contrast, enables a model to attend to information from a different sequence. This is crucial in encoder-decoder architectures like translation or image captioning, where the decoder uses cross-attention to focus on relevant parts of the encoder's output.

**Practical example:** Imagine you're translating "The cat sat on the mat" to French. Self-attention helps the model understand that "cat" relates to "sat" in the English sentence. Cross-attention allows the decoder (generating French) to look back at the relevant English words when producing each French word - it "crosses" between the two sequences.

Both mechanisms are vital for transformer flexibility. Self-attention is fundamental for understanding sequences internally, while cross-attention enables models to relate different sequences effectively, making it essential for tasks requiring integration of multiple data sources (like translation, summarization with context, or vision-language models).

**Learn more:**
- [Cross-Attention Mechanism in Transformers](https://www.geeksforgeeks.org/nlp/cross-attention-mechanism-in-transformers)
- [Self-Attention vs Cross-Attention: From Fundamentals to Applications](https://medium.com/@hexiangnan/self-attention-vs-cross-attention-from-fundamentals-to-applications-4b065285f3f8)

---

### 3. Multi-headed attention - don't understand it

Multi-headed attention is a mechanism that allows the model to focus on different parts of an input sequence simultaneously, which enhances its ability to understand complex relationships within the data.

**How it works:** Instead of computing a single attention score, the process involves creating multiple "heads," each with its own set of learned linear projections for queries, keys, and values. These heads operate in parallel, each capturing different types of dependencies or features within the sequence. The outputs from all heads are then concatenated and linearly transformed to produce the final attention output.

**Why multiple heads are useful:** They enable the model to attend to information from different representation subspaces of the input. Think of it like having multiple experts, each specializing in different aspects:
- One head might focus on short-range dependencies (nearby words)
- Another captures long-range relationships (words far apart in the sentence)
- Another might specialize in syntactic relationships (subject-verb agreement)
- Yet another might focus on semantic relationships (synonyms, related concepts)

**Concrete example:** In the sentence "The animal didn't cross the street because it was too tired," one attention head might focus on "animal" and "it" (coreference), while another head focuses on "cross" and "street" (action-location), and another on "tired" and "didn't cross" (causality). By having multiple heads, the transformer gets a richer, more comprehensive understanding of the input.

This multi-faceted approach makes transformers more effective at capturing complex patterns, especially valuable in tasks like translation, summarization, and question-answering.

**Learn more:**
- [The Math Behind Multi-Head Attention in Transformers](https://towardsdatascience.com/the-math-behind-multi-head-attention-in-transformers-c26cba15f625)
- [Transformers Explained Visually: Multi-head Attention](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

---

### 4. Output matrix - no idea either

The output matrix in transformer attention plays a crucial role in representing the processed information after the attention mechanism has been applied.

**What it contains:** The output matrix contains the weighted sum of the value vectors (V), where the weights are determined by the attention scores derived from the query (Q) and key (K) matrices. This matrix essentially captures how much focus each position in the input sequence has on other positions, allowing the model to integrate relevant contextual information.

**How it works (step by step):**
1. The attention mechanism computes a score for each pair of query and key vectors (using scaled dot-product attention that you already know about)
2. These scores are normalized via softmax to produce attention weights (which sum to 1, as you mentioned with masking)
3. The output matrix is obtained by multiplying these weights with the value vectors
4. This produces a new representation that emphasizes the most relevant parts of the input for each position in the sequence

**Why it matters:** In multi-headed attention, each head produces its own output matrix. These are concatenated and then multiplied by a final output projection matrix (W_o) to combine the insights from all heads into a single representation. This final step allows the model to synthesize information from all the different "attention perspectives" that the multiple heads provide.

This process allows the transformer to dynamically focus on different parts of the input, enabling it to handle long-range dependencies and complex patterns effectively - building directly on the Q, K, V mechanics you already understand.

**Learn more:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)
- [The Transformer Attention Mechanism](https://machinelearningmastery.com/the-transformer-attention-mechanism)

---

## Next Steps

- Review these explanations alongside the original resource
- Try implementing these concepts in code (start with single-head attention, then extend to multi-head)
- Update your learning note with new insights
- Move understood concepts from 'Known Unknown' to 'Known'
- Consider exploring LoRA (Low-Rank Adaptation) to see practical applications of low-rank transformations
