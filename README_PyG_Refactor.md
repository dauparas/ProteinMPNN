# ProteinMPNN PyTorch Geometric Refactoring

This directory contains a refactored implementation of ProteinMPNN using PyTorch Geometric (PyG), providing a cleaner, more maintainable codebase while preserving the core functionality.

## 📁 New Files

- **`protein_mpnn_pyg.py`** - PyTorch Geometric refactored implementation
- **`tutorial_proteinmpnn_pyg.ipynb`** - Comprehensive tutorial notebook
- **`README_PyG_Refactor.md`** - This file

## 🎯 Why Refactor?

### Problems with Original Implementation

The original ProteinMPNN (`protein_mpnn_utils.py`) is a brilliant model, but has some maintainability challenges:

1. **Custom graph operations**: Uses manual `gather_nodes()` and `cat_neighbors_nodes()` functions
2. **Hard to extend**: Monolithic code makes it difficult to experiment with new architectures
3. **Non-standard patterns**: Doesn't follow established GNN design patterns
4. **Limited documentation**: Sparse comments and no type hints

### Benefits of PyTorch Geometric

PyTorch Geometric is the standard library for Graph Neural Networks in PyTorch:

- ✅ **Cleaner code**: Uses `MessagePassing` base class
- ✅ **Standard format**: `edge_index` instead of custom E_idx
- ✅ **Optimized operations**: Leverages sparse GPU kernels
- ✅ **Better documentation**: Type hints and comprehensive docstrings
- ✅ **Easy to extend**: Modular components

## 🏗️ Architecture Comparison

### Original Implementation

```python
class EncLayer(nn.Module):
    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        # Manual neighbor gathering
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(...)

        # Manual message computation
        h_message = self.W3(self.act(self.W2(...)))

        # Manual aggregation
        dh = torch.sum(h_message, -2) / self.scale
        ...
```

### PyG Refactored

```python
class EncLayerPyG(MessagePassing):
    def forward(self, h_V, h_E, edge_index, mask_V, mask_attend):
        # Automatic message passing
        h_message = self.propagate(edge_index, h_V=h_V, h_E=h_E, ...)
        ...

    def message(self, h_V_i, h_V_j, h_E, mask_attend):
        # PyG handles gathering automatically
        h_EV = torch.cat([h_V_i, h_E], dim=-1)
        return self.W3(self.act(self.W2(...)))
        # PyG handles aggregation automatically
```

## 📊 Key Components

### 1. RBF Expansion (`RBFExpansion`)

Converts continuous distances to fixed-size features using Gaussian basis functions.

```python
rbf = RBFExpansion(num_rbf=16, d_min=2.0, d_max=22.0)
features = rbf(distances)  # [N, 1] → [N, 16]
```

### 2. Positional Encodings (`PositionalEncodings`)

Embeds relative sequence position and chain information.

```python
pos_enc = PositionalEncodings(num_embeddings=16)
features = pos_enc(offset, same_chain)  # [N] → [N, 32]
```

### 3. Protein Features (`ProteinFeaturesPyG`)

Constructs k-NN graph from protein backbone coordinates.

```python
features = ProteinFeaturesPyG(edge_features=128, top_k=30, ca_only=True)
edge_features, edge_index = features(X, mask, residue_idx, chain_labels)
```

### 4. Encoder Layer (`EncLayerPyG`)

Message passing layer that updates both nodes and edges.

```python
encoder = EncLayerPyG(num_hidden=128, num_in=256)
h_V_new, h_E_new = encoder(h_V, h_E, edge_index)
```

### 5. Decoder Layer (`DecLayerPyG`)

Autoregressive decoding with masked attention.

```python
decoder = DecLayerPyG(num_hidden=128, num_in=384)
h_V_new = decoder(h_V, h_E, edge_index, mask_attend=mask)
```

### 6. Complete Model (`ProteinMPNNPyG`)

Full encoder-decoder architecture.

```python
model = ProteinMPNNPyG(
    num_letters=21,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    ca_only=True
)
```

## 🚀 Usage

### Installation

```bash
# Install PyTorch Geometric
pip install torch torchvision
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse
```

### Quick Start

```python
from protein_mpnn_pyg import ProteinMPNNPyG
import torch

# Create model
model = ProteinMPNNPyG(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    ca_only=True
)

# Prepare input (CA coordinates)
B, N = 2, 50  # batch_size, seq_length
X = torch.randn(B, N, 3)  # CA coords
S = torch.randint(0, 21, (B, N))  # sequence
mask = torch.ones(B, N)
chain_M = torch.ones(B, N)
residue_idx = torch.arange(N).unsqueeze(0).expand(B, -1)
chain_labels = torch.zeros(B, N, dtype=torch.long)
randn = torch.randn(B, N)

# Forward pass
log_probs = model(X, S, mask, chain_M, residue_idx, chain_labels, randn)
print(log_probs.shape)  # [2, 50, 21]
```

### Tutorial Notebook

See `tutorial_proteinmpnn_pyg.ipynb` for a comprehensive guide including:

- Detailed architecture explanation
- Component-by-component walkthrough
- Visualization examples
- Comparison with original implementation
- Advanced usage patterns

## 📈 Performance Considerations

### Memory

- **Original**: Dense tensor operations
- **PyG**: Sparse graph operations (better for large proteins)

### Speed

- **Small proteins** (<100 residues): Similar performance
- **Large proteins** (>500 residues): PyG is faster due to optimized kernels
- **Batching**: PyG's batching can be more efficient for variable-length proteins

### GPU Utilization

PyG uses optimized CUDA kernels for:
- Sparse matrix multiplication
- Neighbor gathering
- Message aggregation

## 🔄 Migration Guide

### For Users

If you're using the original `protein_mpnn_run.py`:

1. The interface is mostly compatible
2. Replace imports: `from protein_mpnn_pyg import ProteinMPNNPyG`
3. Model weights are NOT compatible (architecture is different)
4. You'll need to retrain or adapt checkpoints

### For Developers

If you're extending ProteinMPNN:

**Original way**:
```python
# Add custom gather operation
def my_gather(features, indices):
    ...
```

**PyG way**:
```python
# Extend MessagePassing
class MyLayer(MessagePassing):
    def message(self, x_i, x_j):
        # PyG handles indexing
        return my_transform(x_i, x_j)
```

## 🛠️ Current Limitations

This refactored version currently has:

1. ✅ **Implemented**:
   - Graph construction (k-NN)
   - RBF distance features
   - Positional encodings
   - Encoder layers (full)
   - Decoder layers (full)
   - Forward pass (basic)
   - CA-only mode

2. ⚠️ **Partial**:
   - Forward pass (simplified autoregressive masking)
   - Training loop

3. ❌ **Not implemented** (TODO):
   - Full backbone mode (N, CA, C, O atoms)
   - Complete sampling method
   - Tied positions logic
   - PSSM constraints
   - All bias/constraint features

## 🤝 Contributing

To extend this implementation:

1. **Add features**: Inherit from `MessagePassing` for new layers
2. **Improve sampling**: Complete the `sample()` method in `ProteinMPNNPyG`
3. **Full backbone**: Extend `ProteinFeaturesPyG` to handle all atoms
4. **Constraints**: Add tied positions, fixed residues, etc.

## 📚 References

- **Original ProteinMPNN**: [Dauparas et al., Science 2022](https://www.science.org/doi/10.1126/science.add2187)
- **PyTorch Geometric**: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)
- **Original code**: [https://github.com/dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)

## 📝 Code Statistics

| Metric | Original | PyG Refactored |
|--------|----------|----------------|
| Total lines | ~1400 | ~800 |
| Functions | Many custom utils | Built-in PyG |
| Type hints | None | Full |
| Docstrings | Minimal | Comprehensive |
| Modularity | Monolithic | High |

## 🎓 Learning Resources

1. **Start here**: `tutorial_proteinmpnn_pyg.ipynb`
2. **PyG basics**: [Official PyG tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
3. **Message Passing**: [Gilmer et al., 2017](https://arxiv.org/abs/1704.01212)
4. **Original paper**: Understanding what ProteinMPNN does

## ⚖️ License

This refactored code maintains the same license as the original ProteinMPNN implementation (MIT License).

## 🙏 Acknowledgments

This refactoring builds upon the excellent work of:
- The ProteinMPNN team at UW Institute for Protein Design
- The PyTorch Geometric developers
- The protein design research community

---

**Questions?** Check the tutorial notebook or open an issue!

**Want to contribute?** Pull requests welcome!

**Using in research?** Please cite the original ProteinMPNN paper.
