"""VENDORED from MGN-Public/meshgraphnet/ml_object/mgn.py.
Patched by scripts/vendor_mgn.py to run without FEniCS.
Do not edit by hand - re-run the vendor script instead.
"""
"""
Mesh Graph Network (MGN) - Geometry-agnostic physics prediction.

Inspired by:

    Learning Mesh-Based Simulation with Graph Networks
    Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia
    https://arxiv.org/abs/2010.03409

Architecture Progression:
========================

1. BasicGCN
   - Node features: [x, y, load] (absolute positions)
   - Edge features: None
   - Limitation: Doesn't transfer to new geometries

2. RelativeGCN  
   - Node features: [x, y, load]
   - Edge features: Implicit (computes x_j - x_i during forward pass)
   - Limitation: Still uses absolute positions

3. MGN (this class!)
   - Node features: [node_type] → learned embedding
   - Edge features: [dx, dy, length] (precomputed, explicit)
   - Global features: [applied_load] (context for all nodes)
   - Goal: Transfer across different geometries!

Features:
- Node type embeddings (learned)
- Edge features: [dx, dy, length] (relative, geometry-agnostic)
- Global features: user-specified metadata (e.g., load magnitude, temperature)

Usage:
    mgn = MGN(hidden_channels=64)
    mgn.fit(train_fems, y_train)
    predictions = mgn.predict(test_fem)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict
from collections import defaultdict
import os
import matplotlib.pyplot as plt

from .model import MGNModel


class MGN:
    """
    Mesh Graph Network with sklearn-like interface.
    
    Unlike GNN which takes (X, y), MGN takes (fem, y) because node features
    are derived from FEM context (boundaries, loads, mesh topology).
    
    Args:
        embedding_dim: Number of "axes of meaning" for node types. 
                       For example:
                       
                            Embedding Space (simplified to 2D)
                    
                        "stress concentrator"
                               ↑
                               |     • hole
                               |     • loaded (high stress here too!)
                               |
                               |              • free
                               |
                               +---------------------------→ "constrained"
                                     • interior
                                             • fixed
                         
                       Instead of one-hot encoding (where "fixed" = [0,1,0,0,0]), we learn a dense vector
                       (where "fixed" = [0.23, -0.15, 0.87, ...]). This lets the network
                       discover relationships between types, e.g., "fixed" and "hole" 
                       might end up with similar embeddings since both are boundaries.
                       Typical values: 8-32. Higher = more expressive but more parameters.      
        hidden_channels: Hidden layer size
        num_layers: Number of message passing layers
        epochs: Maximum training epochs
        learning_rate: Learning rate
        patience: Early stopping patience (-1 for no early stopping)
        global_features: Especially useful when training different FEM setups, like different materials
                        or temperatures. These are set as fem.metadata in the FEM_Object class and are passed
                        here as string (e.g., ["temperature"]).
    
    Example:
        mgn = MGN(hidden_channels=64)
        mgn.fit(train_fems, y_train)
        predictions = mgn.predict(test_fem)
    """


    NODE_TYPE_COLORS = {
        'interior': '#CCCCCC',
        'fixed': '#2196F3',
        'applied_load': '#FF5722',
        'free': '#4CAF50',
        'hole': '#9C27B0',
        'corner': '#FFEB3B',
    }

    def __init__(
            self,
            *,
            embedding_dim: int = 16,
            hidden_channels: int = 64,
            num_layers: int = 20,
            epochs: int = 2000,
            learning_rate: float = 0.01,
            patience: int = -1,
            global_features: List[str] = None,  # E.g., ['load (psi)', 'temperature']
    ):
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience if patience >= 0 else float("inf")
        self.global_features = global_features or []

        # Model and normalization (set during fit)
        self._model = None
        self._y_mean = None
        self._y_std = None
        self._edge_attr_mean = None
        self._edge_attr_std = None
        self._global_mean = None
        self._global_std = None

        # For train_score() - store refs to training FEMs
        self._train_fem = None
        self._y_train = None

        # Node type mapping (built during fit)
        self.node_type_to_id: Dict[str, int] = {}
        self.num_node_types = None

        # Training history
        self.train_losses = []
        self.transductive = False  # MGN is always inductive

        # For mixed precision (optimized)
        self._scaler = None  

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"MGN using: {self.device}")

    def fit(self, fem, y, batch_size=1):
        """
        Train the MGN.
        
        Args:
            fem: FEMObject or List[FEMObject] - training FEM(s)
            y: Target values (e.g., von Mises stress)
            batch_size: Number of FEMs per batch (default=1 for minimal memory)
        
        Returns:
            self
        
        Structure:
            fit()
             ├── _preprocess_fems()                 Turns FEMs into graph components, like node types and global load
             ├── _compute_normalization_stats()     Normalizes, e.g., y -> (y-mean)/std
             ├── _build_model()
             └── _training_loop()
                  └── _train_epoch()
                       └── _train_batch()
        """
        fem_list = fem if isinstance(fem, list) else [fem]
        self._train_fem = fem_list
        self._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    
        # Prepare data
        fem_data = self._preprocess_fems(fem_list)
        self._compute_normalization_stats(fem_data)
        self._build_model()
    
        # Train
        self._training_loop(fem_list, fem_data, batch_size)
    
        return self


    def _preprocess_fems(self, fem_list):
        """Pre-process each FEM into graph components."""
        fem_data = []
        y_offset = 0
        all_node_types = []
    
        for f in fem_list:
            node_types, coords, edge_index, global_features = self._fem_to_graph([f])
            n_nodes = len(coords)
            all_node_types.extend(node_types)
    
            # Pre-compute edge features once per fem
            edge_attr = self._compute_edge_features(coords, edge_index)
    
            fem_data.append({
                'node_types': node_types,
                'coords': coords,
                'edge_index': edge_index,
                'edge_attr': edge_attr,  # Store pre-computed
                'global_features': global_features,
                'y': self._y_train[y_offset:y_offset + n_nodes],
            })
            y_offset += n_nodes
    
        # Build node type mapping from all data
        self._encode_node_types(all_node_types)
        self.num_node_types = len(self.node_type_to_id)
    
        # Pre-encode each FEM's node types
        for fd in fem_data:
            fd['node_type_ids'] = self._encode_node_types(fd['node_types'])
    
        return fem_data


    def _compute_normalization_stats(self, fem_data):
        # Target normalization
        self._y_mean = self._y_train.mean()
        self._y_std = self._y_train.std()
    
        # Edge normalization from ALL FEMs (important for varying geometries!)
        all_edge_attrs = torch.cat([fd['edge_attr'] for fd in fem_data], dim=0)
        self._edge_attr_mean = all_edge_attrs.mean(dim=0)
        self._edge_attr_std = all_edge_attrs.std(dim=0)
    
        # Global normalization
        if self.global_features:
            all_globals = torch.cat([fd['global_features'] for fd in fem_data], dim=0)
            self._global_mean = all_globals.mean(dim=0)
            self._global_std = all_globals.std(dim=0)
    
        # Move to device once
        self._y_mean = self._y_mean.to(self.device)
        self._y_std = self._y_std.to(self.device)
        self._edge_attr_mean = self._edge_attr_mean.to(self.device)
        self._edge_attr_std = self._edge_attr_std.to(self.device)
        if self.global_features:
            self._global_mean = self._global_mean.to(self.device)
            self._global_std = self._global_std.to(self.device)
    
    
    def _build_model(self):
        """Initialize the MGN model."""
        self._model = MGNModel(
            num_node_types=self.num_node_types,
            embedding_dim=self.embedding_dim,
            edge_feature_dim=3,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            global_feature_dim=len(self.global_features),
        ).to(self.device)
    
    
    def _training_loop(self, fem_list, fem_data, batch_size):
        """Main training loop with mini-batching."""
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Enable mixed precision
        self._scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
    
        self._model.train()
        pbar = tqdm(range(self.epochs), desc="Training MGN")
        best_loss = None
        best_state = None
        patience_counter = 0
        self.train_losses = []
    
        for epoch in pbar:
            avg_loss = self._train_epoch(fem_list, fem_data, batch_size, optimizer, loss_fn)
            self.train_losses.append(avg_loss)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
            # Early stopping
            if best_loss is None or (best_loss - avg_loss) / best_loss > 0.001:
                best_loss = avg_loss
                best_state = deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
    
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
        if best_state is not None:
            self._model.load_state_dict(best_state)
    
    
    def _train_epoch(self, fem_list, fem_data, batch_size, optimizer, loss_fn):
        """Train for one epoch, return average loss."""
        epoch_loss = 0.0
        num_batches = 0
    
        # Shuffle FEMs
        indices = list(range(len(fem_data)))
        np.random.shuffle(indices)
    
        # Process in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            loss = self._train_batch(fem_list, fem_data, batch_indices, optimizer, loss_fn)
            epoch_loss += loss
            num_batches += 1

        # Clear cache to lessen memory resources
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
        return epoch_loss / num_batches


    def _train_batch(self, fem_list, fem_data, batch_indices, optimizer, loss_fn):
        """Train on a single batch, return loss."""

        # Combine pre-computed data
        batch_node_ids = torch.cat([fem_data[j]['node_type_ids'] for j in batch_indices]).to(self.device)
        batch_y = torch.cat([fem_data[j]['y'] for j in batch_indices]).to(self.device)
    
        # Combine edge indices with offset
        edge_indices = []
        edge_attrs = []
        node_offset = 0
        for j in batch_indices:
            fd = fem_data[j]
            edge_indices.append(fd['edge_index'] + node_offset)
            edge_attrs.append(fd['edge_attr'])
            node_offset += len(fd['node_types'])
    
        edge_index = torch.cat(edge_indices, dim=1).to(self.device)
        edge_attr = torch.cat(edge_attrs, dim=0).to(self.device)
        edge_attr_norm = (edge_attr - self._edge_attr_mean) / (self._edge_attr_std + 1e-8)
    
        y_norm = (batch_y - self._y_mean) / (self._y_std + 1e-8)
    
        # Global features
        if self.global_features:
            global_features = torch.cat([fem_data[j]['global_features'] for j in batch_indices]).to(self.device)
            global_features_norm = (global_features - self._global_mean) / (self._global_std + 1e-8)
        else:
            global_features_norm = None
    
        # Forward/backward with optional mixed precision
        optimizer.zero_grad()
    
        if self._scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = self._model(batch_node_ids, edge_index, edge_attr_norm, global_features_norm)
                loss = loss_fn(pred, y_norm)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            pred = self._model(batch_node_ids, edge_index, edge_attr_norm, global_features_norm)
            loss = loss_fn(pred, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            optimizer.step()
    
        return loss.item()

    def predict(self, fem):
        """Predict on FEM(s)."""
        if self._model is None:
            raise ValueError("Must call fit() first")
    
        # Handle list by predicting one at a time
        if isinstance(fem, list):
            predictions = []
            for f in fem:
                predictions.append(self.predict(f))
            return np.concatenate(predictions)
    
        # Single FEM from here
        node_types, coords, edge_index, global_features = self._fem_to_graph([fem])
    
        # Encode and compute features
        node_type_ids = self._encode_node_types(node_types).to(self.device)  # <- Add .to(self.device)
        edge_index = edge_index.to(self.device)  # <- Add .to(self.device)
        edge_attr = self._compute_edge_features(coords, edge_index.cpu()).to(self.device)  # <- Add .to(self.device)
    
        # Normalize edge features (stats already on device)
        edge_attr_norm = (edge_attr - self._edge_attr_mean) / (self._edge_attr_std + 1e-8)
    
        # Normalize global features (if any)
        if global_features is not None:
            global_features = global_features.to(self.device)  # <- Add .to(self.device)
            global_features_norm = (global_features - self._global_mean) / (self._global_std + 1e-8)
        else:
            global_features_norm = None
    
        # Predict
        self._model.eval()
        with torch.no_grad():
            pred = self._model(node_type_ids, edge_index, edge_attr_norm, global_features_norm)
            pred = pred * self._y_std + self._y_mean
    
        return pred.cpu().numpy()

    def train_score(self):
        """Return R² on training data."""
        if self._train_fem is None:
            raise ValueError("Must call fit() first")

        y_pred = self.predict(self._train_fem)
        y_actual = self._y_train.cpu().numpy()

        ss_res = ((y_actual - y_pred) ** 2).sum()
        ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    def score(self, fem, y):
        """Return R² score."""
        y_pred = self.predict(fem)
        y = np.array(y).squeeze()

        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    # -------------------------------------------------------------------------
    # Graph construction from FEM
    # -------------------------------------------------------------------------

    def _fem_to_graph(self, fem_list):
        """
        Convert FEMObject(s) to graph components.
        
        Returns:
            (node_types, coords, edge_index, global_features)
        """
        all_node_types = []
        all_coords = []
        all_edge_indices = []
        all_global_features = []
        node_offset = 0

        for fem in fem_list:
            node_types = self._classify_nodes(fem)
            all_node_types.extend(node_types)

            coords = fem.coordinates
            all_coords.append(coords)

            edge_index = fem.edge_index.clone()
            edge_index += node_offset
            all_edge_indices.append(edge_index)

            # Extract specified global features from metadata
            if self.global_features:
                global_vec = []
                for feat_name in self.global_features:
                    value = fem.metadata.get(feat_name)
                    if value is None:
                        raise ValueError(f"Global feature '{feat_name}' not found in FEM metadata. "
                                         f"Available: {list(fem.metadata.keys())}")
                    global_vec.append(float(value))
                all_global_features.extend([global_vec] * len(fem.coordinates))

            node_offset += len(coords)

        coords = np.vstack(all_coords)
        edge_index = torch.cat(all_edge_indices, dim=1)

        if self.global_features:
            global_features = torch.tensor(all_global_features, dtype=torch.float)
        else:
            global_features = None

        return all_node_types, coords, edge_index, global_features


    # PATCHED: node types are precomputed in mgn/graph.py from Abaqus
    # connectivity. Returned as the paper's five strings so that
    # _encode_node_types() produces the same alphabetical map as their
    # released checkpoint: applied_load=0 fixed=1 free=2 hole=3 interior=4
    def _classify_nodes(self, fem):
        from .graph import NODE_TYPE_TO_ID
        id_to_name = dict((v, k) for k, v in NODE_TYPE_TO_ID.items())
        return [id_to_name[int(i)] for i in fem.node_types]

    def _encode_node_types(self, node_types) -> torch.Tensor:
        """Convert node type strings to tensor of integer IDs."""
        # Build mapping if needed (first call during fit)
        if not self.node_type_to_id:
            unique_types = sorted(set(node_types))
            self.node_type_to_id = {t: i for i, t in enumerate(unique_types)}
    
            # Count each type
            from collections import Counter
            counts = Counter(node_types)
    
            print(f"Node types found:")
            for node_type in unique_types:
                print(f"  {node_type}: {counts[node_type]}")
            print(f"  Total: {len(node_types)}")

        # Handle unknown node types at inference time
        unknown_types = set(node_types) - set(self.node_type_to_id.keys())
        if unknown_types:
            print(f"WARNING: Unknown node types encountered: {unknown_types}")
            print(f"  Known types: {list(self.node_type_to_id.keys())}")
            print(f"  Mapping unknown types to 'interior' (or first available type)")
    
            # Fallback to 'interior' if it exists, otherwise first type
            fallback = self.node_type_to_id.get('interior', 0)
    
            ids = [self.node_type_to_id.get(t, fallback) for t in node_types]
        else:
            ids = [self.node_type_to_id[t] for t in node_types]
                
        return torch.tensor(ids, dtype=torch.long)

    def _compute_edge_features(self, coords: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge features from node coordinates.
        
        Edge features: [dx, dy, length]
        - dx, dy: Relative displacement (target - source)
        - length: Euclidean distance
        
        These are RELATIVE features - no absolute positions!
        This is what allows transfer to new geometries.
        """
        sources = edge_index[0].cpu().numpy()
        targets = edge_index[1].cpu().numpy()

        dx = coords[targets, 0] - coords[sources, 0]    # TODO: should we be doing unit vector here?
        dy = coords[targets, 1] - coords[sources, 1]
        length = np.sqrt(dx**2 + dy**2)

        edge_attr = np.stack([dx, dy, length], axis=1)
        return torch.tensor(edge_attr, dtype=torch.float)

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, filepath):
        """Save trained model."""
        if self._model is None:
            raise ValueError("Must call fit() first")

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        save_dict = {
            'model_state_dict': self._model.state_dict(),
            'num_node_types': self.num_node_types,
            'embedding_dim': self.embedding_dim,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'node_type_to_id': self.node_type_to_id,
            'y_mean': self._y_mean.cpu(),
            'y_std': self._y_std.cpu(),
            'edge_attr_mean': self._edge_attr_mean.cpu(),
            'edge_attr_std': self._edge_attr_std.cpu(),
            'global_features': self.global_features,
            'global_mean': self._global_mean.cpu() if self._global_mean is not None else None,
            'global_std': self._global_std.cpu() if self._global_std is not None else None,
            'train_losses': self.train_losses,
        }

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device=None):
        """Load trained model."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        mgn = cls(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_channels=checkpoint['hidden_channels'],
            num_layers=checkpoint['num_layers'],
            learning_rate=checkpoint['learning_rate'],
            global_features=checkpoint.get('global_features', []),
        )

        mgn.device = device
        mgn.num_node_types = checkpoint['num_node_types']
        mgn.node_type_to_id = checkpoint['node_type_to_id']
        mgn._y_mean = checkpoint['y_mean'].to(device)
        mgn._y_std = checkpoint['y_std'].to(device)
        mgn._edge_attr_mean = checkpoint['edge_attr_mean'].to(device)
        mgn._edge_attr_std = checkpoint['edge_attr_std'].to(device)
        mgn.train_losses = checkpoint['train_losses']

        # Globals features are optional
        if checkpoint.get('global_mean') is not None:
            mgn._global_mean = checkpoint['global_mean'].to(device)
            mgn._global_std = checkpoint['global_std'].to(device)

        # Rebuild model
        mgn._model = MGNModel(
            num_node_types=mgn.num_node_types,
            embedding_dim=mgn.embedding_dim,
            edge_feature_dim=3,
            hidden_channels=mgn.hidden_channels,
            num_layers=mgn.num_layers,
            global_feature_dim=len(mgn.global_features),
        ).to(device)
        mgn._model.load_state_dict(checkpoint['model_state_dict'])
        mgn._model.eval()

        print(f"Model loaded from {filepath}")
        return mgn

    def plot_predictions(self, fem, show=True, save_path=None, marker_size=10):
        """
        Plot predicted vs actual values for a solved FEM.
        
        Parameters:
        -----------
        fem : FEMObject or List[FEMObject]
            Solved FEM(s) to compare against
        show : bool
            Whether to display the plot
        save_path : str, optional
            Path to save the figure
        """
        fem_list = fem if isinstance(fem, list) else [fem]
    
        # Get predictions and actuals
        y_pred = self.predict(fem_list)
        y_test = np.concatenate([f.von_mises for f in fem_list])
    
        # Get node types for coloring
        node_types, _, _, _ = self._fem_to_graph(fem_list)
        unique_types = sorted(set(node_types))
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
        # Scatter plot colored by node type
        ax1 = axes[0]
        for node_type in unique_types:
            mask = np.array([t == node_type for t in node_types])
            color = self.NODE_TYPE_COLORS.get(node_type, '#000000')
            ax1.scatter(y_test[mask], y_pred[mask], alpha=0.5, s=marker_size,
                        color=color, label=f'{node_type} ({mask.sum()})')
    
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax1.set_xlabel('Actual (FEM)')
        ax1.set_ylabel('Predicted (MGN)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
    
        # Residuals
        ax2 = axes[1]
        residuals = y_pred - y_test
        for node_type in unique_types:
            mask = np.array([t == node_type for t in node_types])
            color = self.NODE_TYPE_COLORS.get(node_type, '#000000')
            ax2.scatter(y_test[mask], residuals[mask], alpha=0.5, s=marker_size, color=color)
    
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Actual (FEM)')
        ax2.set_ylabel('Residual (Pred - Actual)')
        ax2.grid(True, alpha=0.3)
    
        # R² score
        r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()
        fig.suptitle(f'MGN Predictions (R²={r2:.4f})', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
    
        if show:
            plt.show()
        else:
            return fig

