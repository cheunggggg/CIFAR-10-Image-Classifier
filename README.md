# CIFAR-10 Image Classifier

CNN-based image classifier trained on CIFAR-10 in PyTorch. Implements a LeNet-style architecture with two convolutional blocks followed by three fully connected layers. Covers training on both CPU and GPU, hyperparameter sweeps over batch size and epoch count, and analytical derivations of gradients and cross-entropy loss. Created by ECE 149 at UCLA. FALL 2024

**Course:** ECE 149 — Foundations of Computer Vision, UCLA  
**Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)

---

## Model Architecture

11-layer network operating on 3-channel 32×32 inputs:

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2d | 3 → 6 channels, kernel 5×5 |
| 2 | ReLU | Activation |
| 3 | MaxPool2d | Kernel 2×2 (downsampling) |
| 4 | Conv2d | 6 → 16 channels, kernel 5×5 |
| 5 | ReLU | Activation |
| 6 | MaxPool2d | Kernel 2×2 (downsampling) |
| 7 | Linear | 400 → 120 |
| 8 | ReLU | Activation |
| 9 | Linear | 120 → 84 |
| 10 | ReLU | Activation |
| 11 | Linear | 84 → 10 (class logits) |

The flattened size feeding into the first linear layer is **400** (16 channels × 5 × 5 spatial), derived from two successive conv+pool operations on 32×32 inputs.

> **Note on backpropagation:** ReLU layers require no additional parameter gradients since they have no learnable weights. Their gradient is 1 for non-negative inputs and 0 for negative inputs.

---

## Training

**Loss:** `CrossEntropyLoss`  
**Optimizer:** SGD, `lr=0.001`, `momentum=0.9`  
**Random seed:** `torch.manual_seed(7)`, `np.random.seed(7)`  
**Loss logging:** every 250 mini-batches

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()   # reset gradients each pass
        loss.backward()
        optimizer.step()
```

Gradients are reset with `optimizer.zero_grad()` before each backward pass. Without this, gradients from prior iterations accumulate and corrupt weight updates.

Model weights are saved to `./net.pth` after training.

---

## Experimental Results (GPU, CIFAR-10 test set)

| Batch Size | Epochs | Test Accuracy |
|-----------|--------|---------------|
| 4 | 5 | 59% |
| 4 | 20 | 61% |
| 16 | 5 | 58% |
| 16 | 20 | **65%** |

Larger batch size (16) with more epochs (20) achieves the best accuracy. Accuracy, batch size, and epochs are not strictly positively correlated — batch sizes that are too small introduce excessive gradient noise, while too many epochs risk overfitting.

**Evaluation code:**
```python
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: %d %%' % (correct / total * 100))
```

---

## Key Concepts Covered

### Cross-Entropy Loss (Binary)
Derived from the Bernoulli likelihood of the classifier output:

```
p(y|x) = ŷ^y · (1 - ŷ)^(1-y)

log p(y|x) = y·log(ŷ) + (1-y)·log(1-ŷ)

L(y, ŷ) = -(1/N) Σ [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)]
```

### Overfitting (100% train / 20% test)
Solutions that help: more training data, L2 regularization, validation-based model selection.  
Does **not** help: increasing model size (worsens overfitting).

### Batch Normalization
- **Training:** normalizes each mini-batch with its own mean and variance, introducing noise that acts as a regularizer.
- **Inference:** uses running mean and variance estimated during training; no batch statistics are computed.

### L2 Regularization and Weight Decay
Adding λwᵀw to the loss introduces a `2λw` term in the gradient, causing each weight update to decay the weights by a factor proportional to their magnitude, keeping weights small and reducing overfitting.

### CNN vs. Fully Connected Networks
CNNs outperform fully connected networks on image data due to computational tractability (shared weights, fewer parameters) and explicit hierarchical feature representation (edges → textures → objects).

### Conv Layer Output Size
For `Cin=32`, `Cout=64`, `k=3`, input `C=32, H=64, W=64`:
- No padding, stride 1 → output: `C=64, H=62, W=62`
- Target output `C=64, H=32, W=32` → `stride=3, padding=16`

---

## Dependencies

```
torch
torchvision
numpy
matplotlib
```

Run on **Google Colab** with GPU runtime for full 20-epoch training.
