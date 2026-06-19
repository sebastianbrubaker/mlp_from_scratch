# A Rough Derivation of the Multilayer Perceptron 
---
## Intro
I came across a really exciting theorem: the \textbf{Universal Approximation Theorem}, which roughly states that neural networks can approximate any function to any desired degree of accuracy. I've always enjoyed learning things from the ground up, so before using any modern deep learning frameworks to tackle problems in my field I wanted to build a neural network from scratch. Below, I derive a simple multilayer perceptron, implement it in NumPy, then train and test using the MNIST dataset to produce a model that can classify handwritten digits with $\sim 97 \%$ accuracy.

## Derivation
### 1. Structural Overview
The multilayer perceptron (MLP) can be visualized as a directed acyclic graph (DAG) composed of $L$ vertical layers of neurons, where each neuron in a given layer is connected to every neuron in the next layer. The data enters the input layer, flows forward through the hidden layers, and out through the output layer producing the prediction(s).

### 2. Forward Propagation and Loss
The activation value of a neuron in some layer $`l`$, denoted $`a^{[l]}_{i}`$, is given by some non-linear activation function, $`g^{[l]}(\cdot)`$ on the pre-activation value, $`z^{[l]}_{i}`$, which is given by a weighted sum of the activations in the previous layer plus a bias term, $`b^{[l]}_i`$. Notice, the point-wise nonlinearity is essential or the entire network would collapse to a simple linear function. Consider the two specific cases of the activation at the first, $`l=0`$, and last layer, $`l=L-1`$, in the network. The activation at the first layer $`a^{[0]}_{i}`$ is equal to the input $`x_i`$. The activation at the last layer will depend on the desired behavior of the network; the task at hand is a classification, therefore, the softmax activation function will be used for $`g^{[L-1]}(\cdot)`$ in tandem with the categorical cross entropy loss function. The softmax activation function transforms the elements of a vector into a probability distribution whose elements sum to $1$ and are bounded in $`[0,1]`$. This pairing is perfect because the output of softmax will be interpreted as a probability distribution of prediction confidence over the classes and the categorical cross entropy loss function measures the discrepancy between two probability distributions and punishes accordingly. Thus, the general forward pass for $`l \in \{ 1, 2, \cdots, L-2\}`$ and $`M`$ classes may be expressed in scalar operations as:
```math
$$\begin{aligned}
a^{[0]}_{i} &= x_{i} \\ 
z^{[l]}_{i} &= \sum_{j}w^{[l]}_{ij}a^{[l-1]}_{j} + b^{[l]}_{i} \\
a^{[l]}_{i} &= g^{[l]}(z^{[l]}_{i}) = \text{ReLU}(z^{[l]}_{i}) = \max(0, z^{[l]}_{i}) \\
a^{[L-1]}_{i} &= \text{softmax}(\mathbf{z}^{[L-1]})_i = \frac{\exp({z^{[L-1]}_{i}})}{\sum_{k}{\exp({z^{[L-1]}_{k}})}}\\
\mathcal{L}_{\text{cce}} &= -\sum^{M}_{m} y_{m}\ln(a^{[L-1]}_{m})
\end{aligned}$$
```

Expressing the neural network as scalar operations is a great intuition builder, but it severely under-leverages modern hardware architecture and libraries such as NumPy which can achieve blistering speed by parallelizing vectorized operations. Additionally, modern datasets are often massive and consequently must be to be split into batches for training. Therefore, the notation, implementation, and wall clock performance may be improved by vectorizing the above for a batch size of $N$ as:
```math
$$
\begin{aligned}
\mathbf{A}^{[0]}&= \mathbf{X} \\
\mathbf{Z}^{[l]} &= \mathbf{W}^{[l]}\mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= \mathbf{ReLU}(\mathbf{Z}^{[l]}) \\
\mathbf{A}^{[L-1]} &= \mathbf{softmax}(\mathbf{Z}^{[L-1]}) \\
\mathcal{L}_{\text{cce}} &= -\frac{1}{N}\sum^{N}_{n=1}\sum^{M}_{m=1} y_{mn}\ln(a^{[L-1]}_{mn})
\end{aligned}
$$
```
with matrices:
```math
$$
\begin{aligned}
    \mathbf{A}^{[l]} &\in \mathbb{R}^{n_l \times N}\\
    \mathbf{W}^{[l]} &\in \mathbb{R}^{n_l \times n_{l-1}}\\
    \mathbf{b}^{[l]} &\in \mathbb{R}^{n_l \times 1} \\
\end{aligned}
$$
```
where each column of $`\mathbf{A}^{[l]}`$ corresponds to a training example.

### 3. Backward Propagation and Gradient Descent
At this point, the model can produce predictions and the performance can be measured, but these predictions are completely naive and the model cannot learn. The model must change its parameters with respect to the loss function in order to achieve better performance, i.e. it must learn.

A very elegant method is gradient descent, a calculus-based optimization technique that updates the parameters using the gradient composed of the partial derivatives of the parameters with respect to the loss function. It provides a clean way of determining exactly how much each weight, bias, pre-activation, and activation affects the loss. Notice that the successful implementation of gradient descent relies on differentiation; now, observe that neural networks may be expressed as composite functions and, subsequently, differentiated using the chain rule (given the composing functions are differentiable). In practice, this differentiation is not done by hand, instead it is achieved using a dynamic programming-based numerical technique called automatic differentiation. The reader may find it helpful to visualize backpropagation as walking backwards through the network, determining how much each activation, pre-activation, weight, and bias affects the loss. Let: 
```math
$$\theta = \{\mathbf{W}^{[1]},\mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L-1]},\mathbf{b}^{[L-1]} \}$$
```
denote the parameter set and
```math
$$
\nabla_{\theta}\mathcal{L} = \left\{ \frac{\partial \mathcal{L}_{\text{cce}}}{\partial \mathbf{W}^{[1]}}, \frac{\partial \mathcal{L}_{\text{cce}}}{\partial \mathbf{b}^{[1]}},\cdots, \frac{\partial \mathcal{L}_{\text{cce}}}{\partial \mathbf{W}^{[L-1]}}, \frac{\partial \mathcal{L}_{\text{cce}}}{\partial \mathbf{b}^{[L-1]}} \right\}
$$
```
denote the corresponding gradient.

Thus, the parameter set will be updated according to: $`\theta_{t+1} = \theta_{t} - \eta (\nabla_{\theta}\mathcal{L})_t`$, where $`\eta`$ denotes the learning rate. Note, while there are several types of gradient descent to choose from, mini-batch is used as steps are being taken in the parameter space based on the gradient of a random subset (or batch) of the full training dataset.

The first question when propagating error backwards through the network is "how does the change in some $`z^{[L-1]}_j`$ affect all $`a^{[L-1]}_i`$ and how does that affect $`\mathcal{L}_{\text{cce}}`$?". Posing this question as a derivative yields:
```math
$$
\frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[L-1]}_j} = \sum_i \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a^{[L-1]}_i} \frac{\partial a^{[L-1]}_i}{\partial z^{[L-1]}_j}
$$
```
As it is clear which derivative is being taken the layer indexing will be temporarily dropped to keep the notation clean.

#### 3.1 $\frac{\partial \mathcal{L}_{\text{cce}}}{\partial a_i}$ Derivative of The Loss With Respect to Softmax
```math
$$
\begin{aligned}
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a_i} &= \frac{\partial}{\partial a_i} \left[- \sum_i y_i \ln(a_i) \right] \\
    &= -\frac{y_i}{a_i}
\end{aligned}
$$
```

#### 3.2 $\frac{\partial a_i}{\partial z_j}$ Derivative of Softmax With Respect to The Pre-Activation
##### Case 1: $i = j$
```math
$$
\begin{aligned}
    \frac{\partial a_i}{\partial z_i} &= \frac{\partial}{\partial z_i} \left[ \frac{\exp(z_i)}{\sum_k \exp(z_k)} \right] \\
    &= \frac{\frac{\partial}{\partial z_i} \left[\exp(z_i)\right] \sum_k\exp(z_k) - \exp(z_i) \frac{\partial}{\partial z_i} \left[\sum_k\exp(z_k) \right]}{\left( \sum_k \exp(z_k)\right)^2} \qquad \because \text{Quotient Rule.} \\
    &= \frac{\exp(z_i) \sum_k\exp(z_k) - \exp(z_i)\exp(z_i)}{\left( \sum_k \exp(z_k)\right)^2} \\
    &= \left( \frac{\exp(z_i)}{\sum_k \exp(z_k)} \right) \left( \frac{\sum_k \exp(z_k) - \exp(z_i)}{\sum_k \exp(z_k)} \right) \\
    &= \left( \frac{\exp(z_i)}{\sum_k \exp(z_k)} \right) \left( \frac{\sum_k \exp(z_k)}{\sum_k \exp(z_k)} -  \frac{ \exp(z_i)}{\sum_k \exp(z_k)}\right) \\
    &= a_i (1 - a_i)
\end{aligned}
$$
```

##### Case 2: $i \ne j$
```math
$$
\begin{aligned}
    \frac{\partial a_i}{\partial z_j} &= \frac{\partial}{\partial z_j} \left[ \frac{\exp(z_i)}{\sum_k \exp(z_k)} \right] \\
    &= \frac{\frac{\partial}{\partial z_j} \left[\exp(z_i)\right] \sum_k\exp(z_k) - \exp(z_i) \frac{\partial}{\partial z_j} \left[\sum_k\exp(z_k) \right]}{\left( \sum_k \exp(z_k)\right)^2} \qquad \because \text{Quotient Rule.} \\
    &= \frac{0(\sum_k \exp(z_k)) - \exp(z_i) \exp(z_j)}{\left( \sum_k \exp(z_k)\right)^2} \\
    &= \frac{- \exp(z_i) \exp(z_j)}{\left( \sum_k \exp(z_k)\right)^2} \\
    &= -\left( \frac{\exp(z_i)}{\sum_k \exp(z_k)} \right)\left( \frac{\exp(z_j)}{\sum_k \exp(z_k)} \right) \\
    &= - a_i a_j
\end{aligned}
$$
```
Now combining the above:
```math
$$
\begin{aligned}
    \implies \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z_j} &= \sum_{i \ne j}\left( - \frac{y_i}{a_i} \right)(-a_i a_j) + \left( - \frac{y_j}{a_j} \right)a_j (1 - a_j) \\
    &= \sum_{i \ne j}\frac{y_i a_i a_j}{a_i} + \left( - \frac{y_j a_j (1 - a_j)}{a_j} \right) \\
    &= \sum_{i \ne j} y_i a_j + y_j a_j - y_j \\
    &= (\sum_{i \ne j} y_i + y_j) a_j - y_j  \\
    &= (1)a_j - y_j \qquad \because \mathbf{y} \text{ is one-hot encoded} \\ 
    &= a_j - y_j 
\end{aligned} 
$$
```
Thus, we arrive at a delightfully clean result!

#### 3.3 Backpropagating Further
Now define the error signal $`\delta^{[l]}_i = \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_i}`$ and observe:
#### 3.3 Backpropagating Further
Now define the error signal $`\delta^{[l]}_i = \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_i}`$ and observe:

```math
\begin{aligned}
    \delta^{[L-1]}_i &= a^{[L-1]}_i - y_i \\
    \implies \mathbf{dZ}^{[L-1]} &= \frac{1}{N} (\mathbf{A}^{[L-1]} - \mathbf{Y}) \qquad \because \text{Differentiating the batch loss.}\\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial w^{[l]}_{ij}} &= \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_i} \frac{\partial z^{[l]}_i}{\partial w^{[l]}_{ij}} \\
    &=  \delta^{[l]}_i \frac{\partial}{\partial w^{[l]}_{ij}}\left[w^{[l]}_{i1}a^{[l-1]}_1 + \cdots + w^{[l]}_{ij}a^{[l-1]}_j + \cdots + b^{[l]}_i\right] \\
    &= \delta^{[l]}_i a^{[l-1]}_j \\
    \implies \mathbf{dW}^{[l]} &= \mathbf{dZ}^{[l]} (\mathbf{A}^{[l-1]})^{\top} \qquad \because \text{Vectorizes to an outer product matrix.} \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a^{[l-1]}_{j}} &= \sum_{i} \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_{i}} \frac{\partial z^{[l]}_{i}}{\partial a^{[l-1]}_{j}} \\
    &= \sum_i \delta^{[l]}_i \frac{\partial}{\partial a^{[l-1]}_{j}}\left[w^{[l]}_{i1}a^{[l-1]}_1 + \cdots + w^{[l]}_{ij}a^{[l-1]}_j + \cdots + b^{[l]}_i\right] \\
    &= \sum_i \delta^{[l]}_i w^{[l]}_{ij} \\
    \implies \mathbf{dA}^{[l-1]} &= (\mathbf{W}^{[l]})^{\top} \mathbf{dZ}^{[l]} \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l-1]}_{j}} &= \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a^{[l-1]}_{j}} \frac{\partial a^{[l-1]}_j}{\partial z^{[l-1]}_{j}} \\
    &= \sum_i \delta^{[l]}_i w^{[l]}_{ij} \text{ReLU}'(z^{[l-1]}_{j}) \\
    \text{With: } &\text{ReLU}'(z) := \begin{cases} 1 \quad \text{if} \quad z > 0 \\
    0 \quad \text{if} \quad z \le 0\end{cases} \qquad \because \text{Let ReLU}'(0) = 0\\
    \implies \mathbf{dZ}^{[l-1]} &= ((\mathbf{W}^{[l]})^{\top} \mathbf{dZ}^{[l]}) \odot \mathbf{ReLU}'(\mathbf{Z}^{[l-1]}) = \mathbf{dA}^{[l-1]} \odot \mathbf{ReLU}'(\mathbf{Z}^{[l-1]}) \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial b^{[l]}_i} &=  \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_{i}}  \frac{\partial z^{[l]}_{i}}{\partial b^{[l]}_{i}} \\
    &= \delta^{[l]}_i(1) \\ 
    &= \delta^{[l]}_i \\
    \implies \mathbf{db}^{[l]} &= \sum^{N}_{n=1} \mathbf{dZ}^{[l]}_{:,n}  
\end{aligned}

```math
\begin{aligned}
    \delta^{[L-1]}_i &= a^{[L-1]}_i - y_i \\
    \implies \mathbf{dZ}^{[L-1]} &= \frac{1}{N} (\mathbf{A}^{[L-1]} - \mathbf{Y}) \qquad \cdot 1/N \text{term arises from differentiating the batch loss.}\\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial w^{[l]}_{ij}} &= \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_i} \frac{\partial z^{[l]}_i}{\partial w^{[l]}_{ij}} \\
    &=  \delta^{[l]}_i \frac{\partial}{\partial w^{[l]}_{ij}}\left[w^{[l]}_{i1}a^{[l-1]}_1 + \cdots + w^{[l]}_{ij}a^{[l-1]}_j + \cdots + b^{[l]}_i\right] \\
    &= \delta^{[l]}_i a^{[l-1]}_j \\
    \implies \mathbf{dW}^{[l]} &= \mathbf{dZ}^{[l]} (\mathbf{A}^{[l-1]})^{\top} \qquad \because \text{Computing every } ij \text{ pair is equivalent to an outer product.} \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a^{[l-1]}_{j}} &= \sum_{i} \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_{i}} \frac{\partial z^{[l]}_{i}}{\partial a^{[l-1]}_{j}} \\
    &= \sum_i \delta^{[l]}_i \frac{\partial}{\partial a^{[l-1]}_{j}}\left[w^{[l]}_{i1}a^{[l-1]}_1 + \cdots + w^{[l]}_{ij}a^{[l-1]}_j + \cdots + b^{[l]}_i\right] \\
    &= \sum_i \delta^{[l]}_i w^{[l]}_{ij} \\
    \implies \mathbf{dA}^{[l-1]} &= (\mathbf{W}^{[l]})^{\top} \mathbf{dZ}^{[l]} \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l-1]}_{j}} &= \frac{\partial \mathcal{L}_{\text{cce}}}{\partial a^{[l-1]}_{j}} \frac{\partial a^{[l-1]}_j}{\partial z^{[l-1]}_{j}} \\
    &= \sum_i \delta^{[l]}_i w^{[l]}_{ij} \text{ReLU}'(z^{[l-1]}_{j}) \\
    \text{With: } &\text{ReLU}'(z) := \begin{cases} 1 \quad \text{if} \quad z > 0 \\
    0 \quad \text{if} \quad z \le 0\end{cases} \qquad \cdot \text{Let RelU}'(0) = 0 \text{ despite it being technically undefined.}\\
    \implies \mathbf{dZ}^{[l-1]} &= ((\mathbf{W}^{[l]})^{\top} \mathbf{dZ}^{[l]}) \odot \mathbf{ReLU}'(\mathbf{Z}^{[l-1]}) = \mathbf{dA}^{[l-1]} \mathbf{\odot} \mathbf{ReLU}'(\mathbf{Z}^{[l-1]}) \\
    \frac{\partial \mathcal{L}_{\text{cce}}}{\partial b^{[l]}_i} &=  \frac{\partial \mathcal{L}_{\text{cce}}}{\partial z^{[l]}_{i}}  \frac{\partial z^{[l]}_{i}}{\partial b^{[l]}_{i}} \\
    &= \delta^{[l]}_i(1) \\ 
    &= \delta^{[l]}_i \\
    \implies \mathbf{db}^{[l]} &= \sum^{N}_{n=1} \mathbf{dZ}^{[l]}_{:,n}  
\end{aligned}
```

## Conclusion
The model achieved $`\sim 97 \%`$ accuracy despite a relatively simple architecture! This derivation and implementation has deepened my own understanding and appreciation of neural networks and the research that goes into creating some of the prominent architectures that drive automation, insight, and innovation in today's society.