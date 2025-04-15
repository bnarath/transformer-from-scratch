# Transformer-from-Scratch: English-to-Malayalam Machine Translation version 1

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-ai4bharat%2Fsamanantar-blue)](https://huggingface.co/datasets/ai4bharat/samanantar)

**Executive Summary:** This project implements the Transformer architecture from scratch to address the challenge of English-to-Malayalam (an Indic language) machine translation. It demonstrates a strong understanding of the Transformer's key components and showcases the development of end-to-end neural machine translation (NMT) systems using PyTorch. The project explores character-level translation (a version 2 will be coming up soon with BPETokenizer). It also highlights the ability to design novel architectures in PyTorch, optimize model training on GPUs. Malayalam was chosen for this project as it is my native language.


## Project Goals

- **Deep Dive into Transformers:** 
  Implemented the core Transformer architecture (encoder, decoder, multi-head attention) from scratch, showcasing a deep understanding of these components, without relying on high-level libraries.
  
- **English-to-Malayalam Translation:** 
  Developed a robust machine translation model capable of translating English text into Malayalam, a morphologically rich language with complex word structures.

- **Performance Optimization:** 
  Investigated and implemented techniques to optimize training speed and enhance model efficiency, including leveraging GPU acceleration for training in PyTorch.

- **Monitoring with TensorBoardX:** 
  Used TensorBoardX for effective visualization and monitoring of training progress, model metrics, and performance tuning.

## Technical Deep Dive

### 1. Data Preprocessing

*   **Dataset:** The `ai4bharat/samanantar` dataset from HuggingFace was used, containing approximately 5 million English-Malayalam sentence pairs. This dataset consists of English-to-Indic language pairs, specifically designed for machine translation tasks. For resource efficiency, I’ve limited the dataset to approximately 1 million pairs for training and 100K for validation to reduce  the training cost. (if you have resources, better use the entire data)

## 2. Transformer Architecture
This section explains the components of the Transformer architecture used for this project.

![transformer_architecture](fig/transformer_architecture.png)
*Figure 1: Image courtesy of [Original Transformer Paper](https://arxiv.org/pdf/1706.03762).*  

### Citation

If you use this project or related methods, please cite the [original Transformer paper](https://arxiv.org/pdf/1706.03762):

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}
```

### Key Layers & Features used across Encoder and Decoder

- **Embedding Layer:**
  - The input tokens are converted into 512-dimensional vectors using an embedding layer with learnable weights. The embedding dimension is defined as $d_{model} = 512$.

- **Positional Encoding:**
  - Positional encoding is added to the input embeddings to provide information about the position of tokens in the sequence. 
   
  $\text{PE}(pos, 2i) = \sin\left( \frac{pos}{10000^{2i/d}} \right)$

  $\text{PE}(pos, 2i+1) = \cos\left( \frac{pos}{10000^{2i/d}} \right)$

  Where:
  - $pos$ is the position of the token in the sequence (starting from 0).
  - $i$ is the index of the dimension (for each position in the encoding vector).
  - $d$ is the total dimensionality of the positional encoding (same as the embedding dimension = $512$).


- **Multi-Head Attention:**
  - The attention mechanism allows the model to focus on different parts of the input sequence. This model uses 8 attention heads. It is similar to where to focus on different aspect of language. 

  2 types of attention
  - Self Attention: Used in both Encoder and Decoder
  - Decoder - Encoder Cross Attention: Used in Decoder

- **Attention (Scaled Dot-Product Attention):**
  - The attention mechanism computes a weighted sum of values (V) based on the similarity between the query (Q) and the key (K). The formula for this is:

    $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

    - $Q$: Query matrix (What am I looking for?). 
    - $K$: Key matrix (What do I have). 
    - $V$: VAlue matrix (What do I offer)

      - The size of $Q$, $K$ and $V$ is: $[batch\_size, num\_heads, seq\_len, d_k]$
    
    - The softmax function is applied to ensure the attention weights sum to 1.

- **Masked Attention**

    $\text{Masked Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$

    - The mask is represented as a large negative value (typically $-\infty$) at positions that need to be masked, and 0 at all other positions.

    - Both the encoder and decoder incorporate a padding mask. This is necessary because the input sequences are padded to a fixed length, matching the maximum sequence length. As a result, the model should ignore the padding tokens, ensuring that they neither attend to other tokens nor are attended to.

    - In the decoder, a "no-look-ahead" mask is used. This ensures that each token can only attend to tokens that have occurred before it, effectively masking any tokens that come after it in the sequence.

  
- **Feed-Forward Network:**
  - The encoder consists of a feed-forward network with one hidden layer containing 2048 neurons and uses ReLU as the activation function.
  
- **Layer Normalization:**
  - Layer normalization is applied to stabilize and speed up training, as described in the original Transformer paper. We normalize the layer's activations (neurons). It stabilizes training by ensuring the activations have a consistent scale and distribution, which can improve convergence speed.

  Given an input vector $f(x) = [x_1, x_2, ..., x_d]$ (where  d is the number of features):

  1. **Compute the mean and variance** of the input across the features:
    $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$

    $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$

  2. **Normalize the input**:
    $\hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    Where $\epsilon$  is a small constant added for numerical stability.

  3. **Scale and shift** the normalized output using learnable parameters $\gamma$ (scale) and $\beta$ (shift):
    $y_i = \gamma \hat{x_i} + \beta$


  
- **Residual Connections:**
  - The architecture incorporates residual connections, ensuring that gradients can flow easily through the network during backpropagation, mitigating the issue of vanishing gradients.


### **Training:**
- **Loss Function:**
  - The model uses cross-entropy loss, which is calculated between the predicted token probabilities and the actual target tokens.
  
    - In case of training, english words are tokenized without \<START\> , \<END\>
    - Malayalam words are tokenized with \<START\> but no \<END\>
    - Real labels are tokenized without \<START\> but with \<END\>
    - Cross entropy loss does not include loss from \<PAD\> tokens. 




    The **Cross-Entropy Loss** for a batch of tokens is calculated as the sum of individual losses for each token, excluding padding tokens. The formula for calculating the cross-entropy loss for a single token $t$ in sequence $i$ is:

    Where:
    - $\hat{y}_{i,t}$ is the predicted probability distribution over the vocabulary for the $t$-th token in sentence $i$ (after applying softmax).
    - $y_{i,t}$ is the true class (or the correct token) at position $t$ in sequence $i$.


    For a batch of size $N$, the total loss is the sum of the losses for all tokens in the batch, ignoring padding tokens. Padding tokens are excluded by setting their contribution to 0 in the loss calculation. The final formula for the batch loss is:

    ![batch_loss_formula](fig/batch_loss_formula.png)

    Where:
    - $N$ is the batch size,
    - $T$ is the sequence length (number of tokens in each sentence),
    - $N'$ is the total number of non-padded tokens in the batch,
    - $\mathbb{1}_{\text{not padding}}$ is an indicator function that is 1 if the token is not padding and 0 if it is padding.


- **Optimizer:**
  - The Adam optimizer is used with it's default learning rate to minimize the loss function.
  

## Implementation Details
![implemenation pipeline](fig/implementation.png)


## To Run:
```
# Clone the Repository:
git clone https://github.com/bnarath/transformer-from-scratch.git

# Navigate to the Project Directory
cd transformer-from-scratch

# Setup env
PYENV_VERSION=3.12.7 python -m venv env ; source env/bin/activate  # For Linux/macOS

# Install Package
pip install .

# Run the training using
sh run.sh
```


### Courtesy
I would like to express my gratitude to the following sources for their inspiration and educational content:

- [StatQuest with Josh Starmer's Transformer Series](https://www.youtube.com/watch?v=zxQyTK8quyY)
- [Andrej Karpathy's video on building ChatGPT](https://www.youtube.com/andrejkarpathy)
- [CodeEmporium's Transformer Series](https://www.youtube.com/watch?v=Xg5JG30bYik&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&index=11)

### Contact
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bincynarath/)

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
