"""
AugMGN - the vendored MGN trainer, extended for EA-GNN edge augmentation.

WHY A SUBCLASS AND NOT AN EDIT
    mgn/trainer.py is regenerated from MGN-Public by scripts/vendor_mgn.py and
    says "do not edit by hand". Every change for augmentation lives here
    instead, so re-vendoring can never wipe it, and trainer.py stays the
    paper's code byte for byte.

WITH aug_perc = 0 THIS IS EXACTLY THE BASELINE
    Every override below starts with "if not self.augmented: return super()",
    so a baseline run goes through the vendored code unchanged: 3 edge
    features, pooled normalisation, same model, same checkpoint format.

WHAT CHANGES WITH aug_perc > 0  (augmentation, NO LayerNorm)
    1. edge features   [dx, dy, length]  ->  [dx, dy, length, is_augmented]
    2. edge encoder    first Linear 3 -> 4 inputs. Nothing else in the model
                       changes; the processor is the paper's (no LayerNorm).
    3. normalisation   dx, dy, length are z-scored with SEPARATE statistics
                       for real and augmented edges, and the flag is left as
                       0/1. Pooled statistics would be dominated by the long
                       random edges (~21 in average vs < 1 in for mesh edges)
                       and squash every real edge into a sliver of range.
    4. checkpoint      stores aug_perc, aug_seed, edge_feature_dim and both
                       sets of edge statistics, so predict.py / fig3 rebuild
                       the identical graph and normalisation.

    Mismatches fail LOUDLY: an augmented model handed a plain graph (or the
    reverse) raises, instead of predicting quietly with the wrong graph.
"""

import torch

from .model import MGNModel
from .trainer import MGN


def results_dir_for(aug_perc):
    """results/ for the baseline, results_aug20/ for 20% - never mixed."""
    if not aug_perc or aug_perc <= 0:
        return "results"
    return "results_aug%d" % round(100 * aug_perc)


class AugMGN(MGN):

    def __init__(self, *, aug_perc=0.0, aug_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.aug_perc = float(aug_perc or 0.0)
        self.aug_seed = int(aug_seed or 0)
        self.edge_feature_dim = 4 if self.augmented else 3

        # separate edge statistics (augmented mode only)
        self._real_edge_mean = None
        self._real_edge_std = None
        self._aug_edge_mean = None
        self._aug_edge_std = None

        # set only while predict() runs, so _compute_edge_features knows
        # which edges of the graph being predicted are augmented
        self._active_flag = None

    @property
    def augmented(self):
        return self.aug_perc > 0

    # ------------------------------------------------------------------ utils

    def _flag_of(self, fem):
        """(E,) float tensor, 1 = augmented. Missing flag = all real edges."""
        n_edges = fem.edge_index.shape[1]
        flag = getattr(fem, "edge_flag", None)
        if flag is None:
            return torch.zeros(n_edges)
        flag = torch.as_tensor(flag, dtype=torch.float).reshape(-1).cpu()
        if flag.numel() != n_edges:
            raise ValueError("'%s': edge_flag has %d entries but the graph has "
                             "%d edges" % (getattr(fem, "geometry", "?"),
                                          flag.numel(), n_edges))
        return flag

    def _check_graph(self, fem):
        """Refuse to mix augmented graphs and non-augmented models."""
        has_aug = bool(self._flag_of(fem).sum() > 0)
        name = getattr(fem, "geometry", "?")
        if has_aug and not self.augmented:
            raise ValueError(
                "Graph '%s' contains augmented edges but this model has "
                "aug_perc=0. Use the baseline dataset (data/dataset.pt) or an "
                "augmented checkpoint." % name)
        if self.augmented and not has_aug:
            raise ValueError(
                "This model expects augmented graphs (aug_perc=%g) but '%s' has "
                "none. Build it with mgn.dataset.build_topology(..., "
                "aug_perc=%g) or use data/dataset_aug%d.pt."
                % (self.aug_perc, name, self.aug_perc, round(100 * self.aug_perc)))

    def _normalize_edges(self, edge_attr):
        """[dx,dy,l,flag] raw -> [dx,dy,l] z-scored per edge class, flag kept 0/1."""
        dev = edge_attr.device
        geo, flag = edge_attr[:, :3], edge_attr[:, 3:4]
        real_n = ((geo - self._real_edge_mean.to(dev))
                  / (self._real_edge_std.to(dev) + 1e-8))
        aug_n = ((geo - self._aug_edge_mean.to(dev))
                 / (self._aug_edge_std.to(dev) + 1e-8))
        return torch.cat([torch.where(flag > 0.5, aug_n, real_n), flag], dim=1)

    # --------------------------------------------------------------- training

    def _preprocess_fems(self, fem_list):
        # The vendored code builds [dx, dy, length] for every edge (real and
        # augmented alike - it is just coordinate subtraction).
        fem_data = super()._preprocess_fems(fem_list)
        for fd, fem in zip(fem_data, fem_list):
            self._check_graph(fem)
            if self.augmented:
                flag = self._flag_of(fem).unsqueeze(1)
                fd["edge_attr"] = torch.cat([fd["edge_attr"], flag], dim=1)
        return fem_data

    def _compute_normalization_stats(self, fem_data):
        # y and load statistics: the vendored code, unchanged.
        super()._compute_normalization_stats(fem_data)
        if not self.augmented:
            return

        ea = torch.cat([fd["edge_attr"] for fd in fem_data], dim=0)   # raw, cpu
        geo, flag = ea[:, :3], ea[:, 3]
        real, aug = geo[flag < 0.5], geo[flag > 0.5]
        if len(aug) < 2:
            raise ValueError("augmented model but fewer than 2 augmented edges")
        self._real_edge_mean, self._real_edge_std = real.mean(0), real.std(0)
        self._aug_edge_mean, self._aug_edge_std = aug.mean(0), aug.std(0)

        # Store edge features ALREADY normalised, and make the vendored
        # normalisation step an identity: (x - 0) / (1 + 1e-8) == x in float32.
        # That lets their _train_batch() run untouched.
        for fd in fem_data:
            fd["edge_attr"] = self._normalize_edges(fd["edge_attr"])
        self._edge_attr_mean = torch.zeros(4, device=self.device)
        self._edge_attr_std = torch.ones(4, device=self.device)

    def _build_model(self):
        if not self.augmented:
            return super()._build_model()
        self._model = MGNModel(
            num_node_types=self.num_node_types,
            embedding_dim=self.embedding_dim,
            edge_feature_dim=self.edge_feature_dim,          # 4
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            global_feature_dim=len(self.global_features),
        ).to(self.device)

    # -------------------------------------------------------------- inference

    def _compute_edge_features(self, coords, edge_index):
        edge_attr = super()._compute_edge_features(coords, edge_index)
        if self._active_flag is None:
            return edge_attr            # training path: flag added in _preprocess_fems
        flag = self._active_flag.to(edge_attr.dtype).unsqueeze(1)
        return self._normalize_edges(torch.cat([edge_attr, flag], dim=1))

    def predict(self, fem):
        if isinstance(fem, list):
            return super().predict(fem)          # loops back into predict(f)
        self._check_graph(fem)
        if not self.augmented:
            return super().predict(fem)
        self._active_flag = self._flag_of(fem)
        try:
            return super().predict(fem)
        finally:
            self._active_flag = None

    # --------------------------------------------------------- save and load

    def save(self, filepath):
        super().save(filepath)
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        ck["edge_feature_dim"] = self.edge_feature_dim
        ck["aug_perc"] = self.aug_perc
        ck["aug_seed"] = self.aug_seed
        if self.augmented:
            ck["real_edge_mean"] = self._real_edge_mean.cpu()
            ck["real_edge_std"] = self._real_edge_std.cpu()
            ck["aug_edge_mean"] = self._aug_edge_mean.cpu()
            ck["aug_edge_std"] = self._aug_edge_std.cpu()
        torch.save(ck, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        """Loads augmented checkpoints AND plain ones (incl. the paper's)."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(filepath, map_location=device, weights_only=False)

        if ck.get("edge_feature_dim", 3) == 3:
            # baseline / released checkpoint: the vendored loader, unchanged
            return MGN.load.__func__(cls, filepath, device)

        mgn = cls(
            embedding_dim=ck["embedding_dim"],
            hidden_channels=ck["hidden_channels"],
            num_layers=ck["num_layers"],
            learning_rate=ck["learning_rate"],
            global_features=ck.get("global_features", []),
            aug_perc=ck["aug_perc"],
            aug_seed=ck["aug_seed"],
        )
        mgn.device = device
        mgn.num_node_types = ck["num_node_types"]
        mgn.node_type_to_id = ck["node_type_to_id"]
        mgn._y_mean = ck["y_mean"].to(device)
        mgn._y_std = ck["y_std"].to(device)
        mgn._edge_attr_mean = ck["edge_attr_mean"].to(device)    # zeros(4)
        mgn._edge_attr_std = ck["edge_attr_std"].to(device)      # ones(4)
        mgn._real_edge_mean = ck["real_edge_mean"].to(device)
        mgn._real_edge_std = ck["real_edge_std"].to(device)
        mgn._aug_edge_mean = ck["aug_edge_mean"].to(device)
        mgn._aug_edge_std = ck["aug_edge_std"].to(device)
        mgn.train_losses = ck["train_losses"]
        if ck.get("global_mean") is not None:
            mgn._global_mean = ck["global_mean"].to(device)
            mgn._global_std = ck["global_std"].to(device)

        mgn._model = MGNModel(
            num_node_types=mgn.num_node_types,
            embedding_dim=mgn.embedding_dim,
            edge_feature_dim=4,
            hidden_channels=mgn.hidden_channels,
            num_layers=mgn.num_layers,
            global_feature_dim=len(mgn.global_features),
        ).to(device)
        mgn._model.load_state_dict(ck["model_state_dict"])
        mgn._model.eval()
        print("Model loaded from %s  (edge augmentation %.0f%%)"
              % (filepath, 100 * mgn.aug_perc))
        return mgn
