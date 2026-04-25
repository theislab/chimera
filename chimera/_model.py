"""Chimera v2.4 -- High-level scvi-tools Model API for cross-species integration.

Implements Tasks 7.1-7.6:
  7.1 ChimeraModel class (BaseModelClass + UnsupervisedTrainingMixin)
  7.2 setup_anndata classmethod
  7.3 get_latent_representation
  7.4 get_alpha, get_delta_alpha, get_correction, get_S_g
  7.5 save / load with Chimera-specific buffer verification
  7.6 get_shared_only_expression
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp_mod
import torch
import torch.nn.functional as F
from anndata import AnnData
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin

from chimera._constants import (
    ABLATION_CONFIGS,
    LAMBDA_ANCHOR,
    LAMBDA_AUX,
    LAMBDA_DISP,
    LAMBDA_FIT,
    LAMBDA_MMD,
    LAMBDA_RES,
    N_HIDDEN,
    N_LATENT,
    WARMUP_ADV_EPOCHS,
    WARMUP_FQR_EPOCHS,
    WARMUP_KL_EPOCHS,
)
from chimera._data_utils import get_equal_species_weights, get_sqrt_species_weights
from chimera._module import ChimeraModule
from chimera._training_plan import ChimeraTrainingPlan

logger = logging.getLogger(__name__)


class _SqrtSpeciesDataSplitter(DataSplitter):
    """DataSplitter that injects sqrt-proportional species sampling for training.

    After the standard train/val/test split, the training DataLoader uses a
    :class:`~torch.utils.data.WeightedRandomSampler` so that species are
    sampled proportionally to the square root of their cell counts, correcting
    for imbalanced species representation.
    """

    def __init__(self, *args, cell_weights: np.ndarray, **kwargs):
        self._full_cell_weights = cell_weights
        super().__init__(*args, **kwargs)

    def train_dataloader(self):
        """Create train data loader with sqrt-proportional species sampling."""
        # Subset the full-adata weights to the training indices.
        train_weights = self._full_cell_weights[self.train_idx]
        # Re-normalize to sum to 1 over the training set.
        train_weights = train_weights / train_weights.sum()

        weighted_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=len(self.train_idx),
            replacement=True,
        )
        # scvi-tools' AnnDataLoader uses BatchSampler internally and sets
        # batch_size=None on the underlying PyTorch DataLoader to disable
        # automatic batching.  We replicate this pattern exactly: wrap our
        # WeightedRandomSampler in a BatchSampler, then build a DataLoader
        # with batch_size=None so it yields pre-batched index lists with
        # the correct tensor shapes.
        batch_size = self.data_loader_kwargs.get("batch_size", 128)
        # Match scvi-tools' DataSplitter.train_dataloader, which hard-codes
        # ``drop_last=False``. ``self.drop_last`` does not exist on the parent.
        batch_sampler = BatchSampler(
            weighted_sampler,
            batch_size=batch_size,
            drop_last=False,
        )
        dataset = self.adata_manager.create_torch_dataset(
            indices=self.train_idx,
            load_sparse_tensor=self.load_sparse_tensor,
        )
        return DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

# Registry key for species (uses LABELS_KEY slot since scvi-tools tracks its
# count in summary_stats as n_labels).
_SPECIES_REGISTRY_KEY = REGISTRY_KEYS.LABELS_KEY


class ChimeraModel(UnsupervisedTrainingMixin, BaseModelClass):
    """High-level scvi-tools model for Chimera v2.4 cross-species integration.

    Wraps :class:`~chimera._module.ChimeraModule` with the standard scvi-tools
    model API: ``setup_anndata``, ``train``, ``get_latent_representation``,
    ``save`` / ``load``, plus Chimera-specific getters for alpha, delta_alpha,
    correction, S_g, and shared-only expression.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`setup_anndata`.
    n_latent
        Dimensionality of the latent space.
    n_hidden
        Hidden layer width for encoder / decoder MLPs.
    lambda_aux
        Weight for the shared-only auxiliary reconstruction loss.
    lambda_res
        Weight for the residual sparsity (L1) loss.
    lambda_anchor
        Weight for the alpha anchor loss.
    lambda_fit
        Weight for the fit-quality reward (FQR) loss.
    lambda_disp
        Weight for dispersion regularization.
    use_observed_lib_size
        If True, use observed library sizes; if False, learn them.
    **model_kwargs
        Additional keyword arguments forwarded to :class:`ChimeraModule`.
    """

    _training_plan_cls = ChimeraTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = N_LATENT,
        n_hidden: int = N_HIDDEN,
        lambda_aux: float = LAMBDA_AUX,
        lambda_res: float = LAMBDA_RES,
        lambda_anchor: float = LAMBDA_ANCHOR,
        lambda_fit: float = LAMBDA_FIT,
        lambda_disp: float = LAMBDA_DISP,
        use_observed_lib_size: bool = True,
        **model_kwargs,
    ) -> None:
        super().__init__(adata)

        # -- Extract dimensions from the registered AnnData --
        n_genes = self.summary_stats.n_vars
        n_species = self.summary_stats.n_labels  # species registered as labels
        n_batch = self.summary_stats.n_batch

        # -- Gene features (varm) --
        if "gene_features" not in adata.varm:
            raise ValueError(
                "adata.varm['gene_features'] is required. "
                "Run chimera._data_utils.assemble_gin_features and attach the "
                "result to adata.varm['gene_features'] before calling setup_anndata."
            )
        gene_features = torch.tensor(
            np.array(adata.varm["gene_features"]), dtype=torch.float32
        )

        # -- Conservation prior (var) --
        if "conservation_prior" not in adata.var.columns:
            raise ValueError(
                "adata.var['conservation_prior'] is required. "
                "Attach the conservation prior (values in [0.05, 0.95]) to "
                "adata.var['conservation_prior'] before calling setup_anndata."
            )
        conservation_prior = torch.tensor(
            adata.var["conservation_prior"].values, dtype=torch.float32
        )

        # -- p_null (uns) --
        if "p_null" not in adata.uns:
            raise ValueError(
                "adata.uns['p_null'] is required. "
                "Run chimera._data_utils.precompute_p_null and attach the "
                "result to adata.uns['p_null'] before calling setup_anndata."
            )
        p_null_np = adata.uns["p_null"]
        p_null = torch.tensor(np.array(p_null_np), dtype=torch.float32)

        # -- batch_to_species_idx (uns) --
        if "batch_to_species_idx" not in adata.uns:
            raise ValueError(
                "adata.uns['batch_to_species_idx'] is required. "
                "This is normally computed by setup_anndata."
            )
        batch_to_species_idx = torch.tensor(
            adata.uns["batch_to_species_idx"], dtype=torch.long
        )

        # -- Construct the module --
        self.module = ChimeraModule(
            n_genes=n_genes,
            n_species=n_species,
            n_batch=n_batch,
            n_latent=n_latent,
            n_hidden=n_hidden,
            gene_features=gene_features,
            conservation_prior=conservation_prior,
            p_null=p_null,
            batch_to_species_idx=batch_to_species_idx,
            use_observed_lib_size=use_observed_lib_size,
            lambda_aux=lambda_aux,
            lambda_res=lambda_res,
            lambda_anchor=lambda_anchor,
            lambda_fit=lambda_fit,
            lambda_disp=lambda_disp,
            **model_kwargs,
        )

        # ChimeraModule extends nn.Module directly (not scvi BaseModuleClass),
        # so we add the on_load callback that scvi-tools BaseModelClass.load
        # expects.
        if not hasattr(self.module, "on_load"):
            self.module.on_load = lambda model, **kw: None

        self.n_latent = n_latent
        self.use_observed_lib_size = use_observed_lib_size

        # Precompute sqrt-proportional species sampling weights (Task 3.9).
        self._sqrt_species_weights = get_sqrt_species_weights(adata)

        self._model_summary_string = (
            f"ChimeraModel v2.4 | "
            f"Genes: {n_genes}, Species: {n_species}, Batches: {n_batch}, "
            f"Latent: {n_latent}, Hidden: {n_hidden}"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------
    # Task 7.2: setup_anndata
    # ------------------------------------------------------------------

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        species_key: str = "species",
        batch_key: str = "batch",
        layer: str | None = None,
        celltype_key: str | None = None,
        **kwargs,
    ) -> None:
        """Register an AnnData object for use with ChimeraModel.

        Parameters
        ----------
        adata
            AnnData with raw UMI counts in ``.X`` (or the specified layer).
            Must also have:
            - ``adata.varm['gene_features']``: per-gene feature matrix
            - ``adata.var['conservation_prior']``: per-gene conservation prior
            - ``adata.uns['p_null']``: null expression proportions ``(K, G)``
        species_key
            Column in ``adata.obs`` identifying species.
        batch_key
            Column in ``adata.obs`` identifying batches.
        layer
            Layer in ``adata.layers`` to use as count data. If None, uses ``.X``.
        celltype_key
            Optional column in ``adata.obs`` with cell-type labels. When provided
            AND the module is configured with ``use_celltype_conditional_mmd=True``,
            MMD is evaluated within each cell type (species are matched per
            cell type) rather than globally. This targets E8.4 (within-celltype
            species mixing) directly. Cell type labels are NOT fed to the encoder
            or to either decoder pathway — they influence only the MMD loss
            signal. If None, the global MMD is used.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`AnnDataManager.register_fields`.
        """
        # Capture setup method args FIRST, before any local variables are
        # created, to avoid capturing unpickleable objects (like imports).
        setup_method_args = cls._get_setup_method_args(**locals())

        # -- Validate Chimera-specific fields exist --
        if "gene_features" not in adata.varm:
            raise ValueError(
                "adata.varm['gene_features'] must be set before calling "
                "setup_anndata. Use chimera._data_utils.assemble_gin_features."
            )
        if "conservation_prior" not in adata.var.columns:
            raise ValueError(
                "adata.var['conservation_prior'] must be set before calling "
                "setup_anndata."
            )
        if "p_null" not in adata.uns:
            raise ValueError(
                "adata.uns['p_null'] must be set before calling setup_anndata. "
                "Use chimera._data_utils.precompute_p_null."
            )

        # -- Compute batch_to_species_idx --
        # Validate that each batch maps to exactly one species.
        batch_col = adata.obs[batch_key]
        species_col = adata.obs[species_key]

        batch_categories = batch_col.cat.categories if hasattr(batch_col, "cat") else sorted(batch_col.unique())
        species_categories = species_col.cat.categories if hasattr(species_col, "cat") else sorted(species_col.unique())

        # Build species -> integer index mapping (sorted)
        species_to_idx = {s: i for i, s in enumerate(sorted(species_categories))}

        batch_to_species = {}
        for b in batch_categories:
            mask = batch_col == b
            species_for_batch = species_col[mask].unique()
            if len(species_for_batch) != 1:
                raise ValueError(
                    f"Batch '{b}' maps to {len(species_for_batch)} species "
                    f"({list(species_for_batch)}). Each batch must map to "
                    f"exactly one species."
                )
            batch_to_species[b] = species_for_batch[0]

        # We need to map batch integer codes (as they will be after
        # CategoricalObsField registration) to species integer codes.
        # CategoricalObsField sorts categories, so the integer codes follow
        # the sorted order of unique values.
        sorted_batches = sorted(batch_categories)
        sorted_species = sorted(species_categories)
        species_idx_map = {s: i for i, s in enumerate(sorted_species)}

        batch_to_species_idx = np.array(
            [species_idx_map[batch_to_species[b]] for b in sorted_batches],
            dtype=np.int64,
        )
        adata.uns["batch_to_species_idx"] = batch_to_species_idx

        # -- Compute per-batch library size statistics --
        # These are used by the learned library size encoder (local_l_mean,
        # local_l_var per batch).
        X = adata.X if layer is None else adata.layers[layer]
        if sp_mod.issparse(X):
            lib_sizes = np.asarray(X.sum(axis=1)).ravel()
        else:
            lib_sizes = np.asarray(X, dtype=np.float64).sum(axis=1)

        log_lib = np.log(lib_sizes + 1e-8)
        local_l_mean = np.zeros(adata.n_obs)
        local_l_var = np.ones(adata.n_obs)

        for b in sorted_batches:
            mask = (batch_col == b).values
            if mask.sum() > 0:
                local_l_mean[mask] = log_lib[mask].mean()
                local_l_var[mask] = log_lib[mask].var() + 1e-8

        adata.uns["local_l_mean"] = local_l_mean
        adata.uns["local_l_var"] = local_l_var

        # -- Register fields with scvi-tools --
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Species goes into LABELS_KEY so that n_labels = n_species
            CategoricalObsField(_SPECIES_REGISTRY_KEY, species_key),
        ]
        # Optionally register a cell-type field for celltype-conditional MMD.
        # The key ``"celltype"`` is read from the tensors dict inside
        # ``ChimeraModule.forward`` when conditional MMD is active. The
        # encoder / decoders never see this field — it is exclusively a
        # training-signal input for MMD grouping.
        if celltype_key is not None:
            if celltype_key not in adata.obs.columns:
                raise ValueError(
                    f"celltype_key={celltype_key!r} is not a column in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )
            anndata_fields.append(CategoricalObsField("celltype", celltype_key))
        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # Task 7.1 (continued): train override
    # ------------------------------------------------------------------

    def train(
        self,
        max_epochs: int | None = 400,
        batch_size: int = 256,
        early_stopping: bool = True,
        plan_kwargs: dict | None = None,
        datasplitter_kwargs: dict | None = None,
        sqrt_species_sampling: bool = True,
        equal_species_sampling: bool = False,
        **kwargs,
    ):
        """Train the Chimera model.

        Overrides the default ``train`` to supply Chimera-specific defaults
        for the training plan and schedule, and to inject sqrt-proportional
        species sampling into the training DataLoader.

        Parameters
        ----------
        max_epochs
            Maximum number of training epochs.
        batch_size
            Minibatch size.
        early_stopping
            Whether to use early stopping (via composite convergence).
        plan_kwargs
            Additional keyword arguments for :class:`ChimeraTrainingPlan`.
        datasplitter_kwargs
            Additional keyword arguments for the data splitter.
        sqrt_species_sampling
            If True (default), use sqrt-proportional species sampling for the
            training DataLoader. This upweights underrepresented species so
            that each species' effective sampling probability is proportional
            to the square root of its cell count, reducing dominance by the
            largest species.
        equal_species_sampling
            If True, overrides ``sqrt_species_sampling`` and uses per-species
            equal weighting so each species contributes 1/S of sampling
            probability. Tightens per-batch species distribution for MMD
            and reduces seed-variance in integration on skewed datasets.
            Default False.
        **kwargs
            Additional keyword arguments forwarded to the parent ``train``.
        """
        plan_kwargs = plan_kwargs or {}
        plan_kwargs.setdefault("warmup_kl_epochs", WARMUP_KL_EPOCHS)
        plan_kwargs.setdefault("warmup_fqr_epochs", WARMUP_FQR_EPOCHS)
        plan_kwargs.setdefault("warmup_adversarial_epochs", WARMUP_ADV_EPOCHS)

        # Override early stopping monitor: we log "validation_loss" not "elbo_validation"
        kwargs.setdefault("early_stopping_monitor", "validation_loss")

        # Resolve which species-weighted sampling scheme to use.
        # equal_species_sampling takes precedence when requested.
        use_weighted = sqrt_species_sampling or equal_species_sampling
        if equal_species_sampling:
            _weights = get_equal_species_weights(self.adata)
        else:
            _weights = self._sqrt_species_weights

        if use_weighted:
            # Inject a custom DataSplitter that uses a WeightedRandomSampler
            # with per-species weights. scvi-tools 1.1.2 builds
            # the splitter internally when adata is registered (passing
            # ``data_module=`` is forbidden in that case), so we hot-swap
            # ``self._data_splitter_cls`` for the duration of this call and
            # hand ``cell_weights`` through ``datasplitter_kwargs``.
            datasplitter_kwargs = dict(datasplitter_kwargs or {})
            datasplitter_kwargs["cell_weights"] = _weights
            original_cls = self._data_splitter_cls
            self._data_splitter_cls = _SqrtSpeciesDataSplitter
            try:
                return super().train(
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    early_stopping=early_stopping,
                    plan_kwargs=plan_kwargs,
                    datasplitter_kwargs=datasplitter_kwargs,
                    **kwargs,
                )
            finally:
                self._data_splitter_cls = original_cls
        else:
            return super().train(
                max_epochs=max_epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                plan_kwargs=plan_kwargs,
                datasplitter_kwargs=datasplitter_kwargs,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # Task 7.3: get_latent_representation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData to use. If None, uses the AnnData registered at init.
        indices
            Cell indices to use. If None, uses all cells.
        batch_size
            Minibatch size for the DataLoader.
        return_dist
            If True, returns ``(z_mean, z_logvar)`` tuple instead of just
            ``z_mean``.

        Returns
        -------
        z_mean or (z_mean, z_logvar)
            Latent representation(s) as numpy array(s) of shape ``(N, n_latent)``.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size or 128, shuffle=False
        )

        self.module.eval()
        z_means = []
        z_logvars = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_idx = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
            species_idx = tensors[_SPECIES_REGISTRY_KEY].squeeze(-1).long()

            # Reconstruct encoder input: cat([x, species_oh, batch_oh])
            species_oh = F.one_hot(species_idx, self.module.n_species).float()
            batch_oh = F.one_hot(batch_idx, self.module.n_batch).float()
            enc_input = self.module._encoder_input(x, species_oh, batch_oh)

            h_enc = self.module.encoder(enc_input)
            z_mean = self.module.z_mean_linear(h_enc)
            z_logvar = self.module.z_logvar_linear(h_enc)

            z_means.append(z_mean.cpu().numpy())
            if return_dist:
                z_logvars.append(z_logvar.cpu().numpy())

        z_means_arr = np.concatenate(z_means, axis=0)
        if return_dist:
            z_logvars_arr = np.concatenate(z_logvars, axis=0)
            return z_means_arr, z_logvars_arr
        return z_means_arr

    # ------------------------------------------------------------------
    # Task 7.4: get_alpha, get_delta_alpha, get_correction, get_S_g
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_alpha(self) -> np.ndarray:
        """Return the current per-gene integration coefficient alpha.

        .. note:: **Caveat 31 (multi-domain proteins):** Alpha is per-gene,
           not per-domain. For multi-domain proteins with one conserved and
           one variable domain (e.g., immunoglobulins, receptor tyrosine
           kinases), alpha reflects the average, not the domain-specific
           conservation. Interpret alpha with caution for known multi-domain
           proteins.

        Returns
        -------
        np.ndarray
            Alpha values of shape ``(G,)`` in ``(0, 1)``.
        """
        self.module.eval()
        if hasattr(self.module, "per_gene_correction"):
            # Config M: per-gene nn.Parameter alpha (no GIN)
            alpha = torch.sigmoid(
                torch.logit(self.module.conservation_prior)
                + self.module.per_gene_correction
            )
        else:
            alpha, _ = self.module.gin(
                self.module.gene_features, self.module.conservation_prior
            )
        return alpha.cpu().numpy()

    @torch.no_grad()
    def get_delta_alpha(self) -> np.ndarray:
        """Return delta_alpha = alpha - conservation_prior per gene.

        Positive values indicate the model has learned the gene is more
        conserved than the prior suggests; negative values indicate less
        conserved.

        .. note:: **Caveat 31 (multi-domain proteins):** Alpha is per-gene,
           not per-domain. For multi-domain proteins with one conserved and
           one variable domain, delta_alpha reflects the average shift, not
           domain-specific conservation changes. See :meth:`get_alpha`.

        .. note:: **Caveat 32 (linear residual underfit):** The residual head
           is a single linear layer. It captures per-species baseline shifts
           but NOT nonlinear species x cell-state interactions (e.g., a gene
           upregulated in mouse hepatocytes but downregulated in mouse T cells
           relative to human). For such genes, alpha is pushed to the floor
           even when only the interaction (not the baseline) is species-
           specific. Compute per-species residual variance at fixed cell type
           to identify genes with unmodeled interactions and unreliable alpha.

        Returns
        -------
        np.ndarray
            Delta alpha values of shape ``(G,)``.
        """
        alpha = self.get_alpha()
        prior = self.module.conservation_prior.cpu().numpy()
        return alpha - prior

    @torch.no_grad()
    def get_correction(self) -> np.ndarray:
        """Return the raw GIN correction output per gene.

        Returns
        -------
        np.ndarray
            Correction values of shape ``(G,)``.
        """
        self.module.eval()
        if hasattr(self.module, "per_gene_correction"):
            # Config M: the correction *is* the per-gene parameter
            return self.module.per_gene_correction.detach().cpu().numpy()
        _, correction = self.module.gin(
            self.module.gene_features, self.module.conservation_prior
        )
        return correction.cpu().numpy()

    @torch.no_grad()
    def get_S_g(self) -> np.ndarray:
        """Return the S_g exponential moving average buffer.

        Returns
        -------
        np.ndarray
            S_g EMA values of shape ``(G,)``.
        """
        return self.module.S_g_ema.cpu().numpy()

    # ------------------------------------------------------------------
    # Task 7.5: save / load
    # ------------------------------------------------------------------

    def save(
        self,
        dir_path: str,
        prefix: str | None = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **kwargs,
    ) -> None:
        """Save the model, including Chimera-specific artifacts.

        Saves ``gene_features_scaler_`` as a pickle alongside the standard
        scvi-tools model checkpoint.

        Parameters
        ----------
        dir_path
            Directory to save to.
        prefix
            Optional filename prefix.
        overwrite
            Whether to overwrite an existing directory.
        save_anndata
            Whether to also save the AnnData.
        **kwargs
            Additional keyword arguments forwarded to the parent ``save``.
        """
        super().save(
            dir_path,
            prefix=prefix,
            overwrite=overwrite,
            save_anndata=save_anndata,
            **kwargs,
        )

        # Save gene_features_scaler_ if it exists
        file_name_prefix = prefix or ""
        if hasattr(self, "gene_features_scaler_"):
            scaler_path = os.path.join(
                dir_path, f"{file_name_prefix}gene_features_scaler.pkl"
            )
            with open(scaler_path, "wb") as f:
                pickle.dump(self.gene_features_scaler_, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: AnnData | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        prefix: str | None = None,
        backup_url: str | None = None,
        **kwargs,
    ):
        """Load a saved ChimeraModel and verify Chimera-specific buffers.

        Parameters
        ----------
        dir_path
            Path to saved model directory.
        adata
            AnnData to associate with the loaded model.
        accelerator
            Accelerator for model placement.
        device
            Device for model placement.
        prefix
            Filename prefix used during save.
        backup_url
            Backup URL for downloading saved files.
        **kwargs
            Additional keyword arguments forwarded to the parent ``load``.

        Returns
        -------
        ChimeraModel
            Loaded model instance.
        """
        model = super().load(
            dir_path,
            adata=adata,
            accelerator=accelerator,
            device=device,
            prefix=prefix,
            backup_url=backup_url,
            **kwargs,
        )

        # Restore gene_features_scaler_ if it was saved
        file_name_prefix = prefix or ""
        scaler_path = os.path.join(
            dir_path, f"{file_name_prefix}gene_features_scaler.pkl"
        )
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                model.gene_features_scaler_ = pickle.load(f)

        # Verify that all Chimera-specific buffers survived the round-trip
        required_buffers = [
            "gene_features",
            "conservation_prior",
            "p_null",
            "S_g_ema",
        ]
        for buf_name in required_buffers:
            if not hasattr(model.module, buf_name):
                raise RuntimeError(
                    f"Chimera-specific buffer '{buf_name}' is missing from "
                    f"the loaded module. The checkpoint may be corrupted."
                )
            buf = getattr(model.module, buf_name)
            if buf is None:
                raise RuntimeError(
                    f"Chimera-specific buffer '{buf_name}' is None in the "
                    f"loaded module."
                )

        # Verify batch_to_species_idx on batch_offset submodule
        if not hasattr(model.module.batch_offset, "batch_to_species_idx"):
            raise RuntimeError(
                "batch_to_species_idx buffer is missing from the "
                "batch_offset submodule."
            )

        return model

    # ------------------------------------------------------------------
    # Task 7.6: get_shared_only_expression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_shared_only_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        library_size: float | str | None = None,
    ) -> np.ndarray:
        """Return the shared-only reconstructed expression for each cell.

        The shared-only expression ``mu_shared_only`` is the reconstruction
        using only the shared decoder head (plus batch offset), without the
        species-specific residual.

        Parameters
        ----------
        adata
            AnnData to use. If None, uses the AnnData registered at init.
        indices
            Cell indices to use. If None, uses all cells.
        batch_size
            Minibatch size for the DataLoader.
        library_size
            Library size normalization. If ``None``, uses each cell's observed
            library size. If a float, scales all cells to that total count. If
            ``"latent"``, returns log-rate-scale values (no library size
            multiplication).

        Returns
        -------
        np.ndarray
            Shared-only expression of shape ``(N, G)``.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size or 128, shuffle=False
        )

        self.module.eval()
        results = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_idx = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
            species_idx = tensors[_SPECIES_REGISTRY_KEY].squeeze(-1).long()

            # Encode
            species_oh = F.one_hot(species_idx, self.module.n_species).float()
            batch_oh = F.one_hot(batch_idx, self.module.n_batch).float()
            enc_input = self.module._encoder_input(x, species_oh, batch_oh)
            h_enc = self.module.encoder(enc_input)
            z_mean = self.module.z_mean_linear(h_enc)

            # Decode (shared only, using z_mean for deterministic output)
            h = self.module.decoder_hidden(z_mean)
            log_rate_shared = self.module.output_shared(h)

            # Add batch offset
            constrained_offset = self.module.batch_offset()
            log_rate_batch = constrained_offset[batch_idx]
            log_rate_total = log_rate_shared + log_rate_batch

            if library_size == "latent":
                # Return log-rate without library size
                results.append(log_rate_total.cpu().numpy())
            else:
                if library_size is not None:
                    lib = float(library_size)
                else:
                    # Use observed library size
                    lib = x.sum(-1, keepdim=True)

                mu_shared = torch.exp(log_rate_total) * lib
                results.append(mu_shared.cpu().numpy())

        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Task 7.7: from_config factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        adata: AnnData,
        config_name: str,
        n_latent: int = N_LATENT,
        n_hidden: int = N_HIDDEN,
        **overrides,
    ) -> "ChimeraModel":
        """Create a ChimeraModel from a named ablation configuration.

        Looks up ``config_name`` in :data:`~chimera._constants.ABLATION_CONFIGS`,
        extracts the loss-weight kwargs (lambda_aux/res/anchor/fit), constructs
        the model, then applies any meta-key flags (freeze_alpha, random_gene_features,
        no_esm2, per_gene_alpha, fqr_null_variant, unified_anchor_lambda).

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`setup_anndata`.
        config_name
            Name of the ablation config (e.g. ``"A"``, ``"B_prime"``, ``"M"``).
        n_latent
            Dimensionality of the latent space.
        n_hidden
            Hidden layer width for encoder / decoder MLPs.
        **overrides
            Additional keyword arguments that override config values and are
            forwarded to the :class:`ChimeraModel` constructor.

        Returns
        -------
        ChimeraModel
            Model instance with the specified ablation configuration applied.

        Raises
        ------
        KeyError
            If ``config_name`` is not found in ``ABLATION_CONFIGS``.
        """
        if config_name not in ABLATION_CONFIGS:
            raise KeyError(
                f"Unknown ablation config '{config_name}'. "
                f"Available: {sorted(ABLATION_CONFIGS.keys())}"
            )

        cfg = dict(ABLATION_CONFIGS[config_name])  # shallow copy

        # Pop meta-keys before passing to constructor
        freeze_alpha = cfg.pop("freeze_alpha", None)
        random_gene_features = cfg.pop("random_gene_features", False)
        no_esm2 = cfg.pop("no_esm2", False)
        per_gene_alpha = cfg.pop("per_gene_alpha", False)
        fqr_null_variant = cfg.pop("fqr_null_variant", False)
        unified_anchor_lambda = cfg.pop("unified_anchor_lambda", None)
        cfg.pop("note", None)

        # Apply overrides (these take priority over config values)
        cfg.update(overrides)

        # Construct the model with loss-weight kwargs
        model = cls(
            adata,
            n_latent=n_latent,
            n_hidden=n_hidden,
            **cfg,
        )

        module = model.module

        # --- Apply meta-key flags ---

        # freeze_alpha: disable GIN learning
        if freeze_alpha is not None:
            if freeze_alpha == "prior":
                # Zero the GIN correction so alpha = conservation_prior
                with torch.no_grad():
                    for param in module.gin.parameters():
                        param.zero_()
                for param in module.gin.parameters():
                    param.requires_grad_(False)
            else:
                # freeze_alpha is a float target value for alpha
                target_alpha = float(freeze_alpha)
                # Compute the correction needed: correction = logit(target) - logit(prior)
                prior_logit = torch.logit(module.conservation_prior)
                target_logit = torch.logit(
                    torch.tensor(target_alpha).clamp(0.01, 0.99)
                )
                needed_correction = target_logit - prior_logit  # (G,)

                # Set the GIN last layer to produce this correction for any input:
                # The last layer is Linear(n_hidden, 1). We set weight=0, bias=correction.
                # But bias is scalar (1,), so we need a different approach:
                # Zero all GIN weights, then set the last layer bias to the mean correction.
                # Actually, since we freeze gradients, we can just override alpha directly.
                # Better approach: zero the GIN weights and set bias to needed_correction mean,
                # BUT the correction_mlp outputs (G,) via broadcasting over genes.
                # Each gene gets the same correction from bias. For a uniform target alpha,
                # this works. For per-gene targets we'd need a different approach.
                with torch.no_grad():
                    # Zero all weights so output = bias regardless of input
                    for param in module.gin.correction_mlp.parameters():
                        param.zero_()
                    # Set last-layer bias to produce the needed correction
                    # Since the MLP processes each gene independently and ReLU(0)=0,
                    # with zeroed weights the hidden layer output is 0, so
                    # correction = W2 * 0 + b2 = b2. Set b2 = average needed correction.
                    # For a single target_alpha the correction is uniform.
                    module.gin.correction_mlp[-1].bias.data.fill_(
                        needed_correction.mean().item()
                    )
                for param in module.gin.parameters():
                    param.requires_grad_(False)

        # random_gene_features: replace gene_features buffer with random N(0,1)
        if random_gene_features:
            with torch.no_grad():
                module.gene_features.normal_(0.0, 1.0)

        # no_esm2: zero the first 64 dims of gene_features (ESM-2 PCA columns)
        if no_esm2:
            with torch.no_grad():
                n_esm2_dims = min(64, module.gene_features.shape[1])
                module.gene_features[:, :n_esm2_dims] = 0.0

        # per_gene_alpha: add per-gene correction parameter, disable GIN
        if per_gene_alpha:
            n_genes = module.n_genes
            module.per_gene_correction = torch.nn.Parameter(
                torch.zeros(n_genes)
            )
            # Disable GIN gradients (per-gene parameter replaces it)
            for param in module.gin.parameters():
                param.requires_grad_(False)
            module._use_per_gene_alpha = True

        # fqr_null_variant: flag for training plan to zero S_g signal
        if fqr_null_variant:
            module._fqr_null_variant = True

        # unified_anchor_lambda: enable unified anchor mode
        if unified_anchor_lambda is not None:
            module.use_unified_anchor = True
            module.unified_anchor_lambda = float(unified_anchor_lambda)
            module.unified_anchor_c = 1.0

        logger.info(
            f"Created ChimeraModel from config '{config_name}'"
            f"{' with overrides' if overrides else ''}"
        )

        return model

    # ------------------------------------------------------------------
    # Task 7.6b: get_expression (full model prediction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        library_size: float | str | None = None,
    ) -> np.ndarray:
        """Return the full reconstructed expression (shared + residual) for each cell.

        This is the counterpart to :meth:`get_shared_only_expression`: it returns
        ``mu_full`` which includes both the shared decoder output and the
        species-specific residual, weighted by ``(1 - alpha)``.

        Uses ``z_mean`` (deterministic, not reparameterized) for inference.

        Parameters
        ----------
        adata
            AnnData to use. If None, uses the AnnData registered at init.
        indices
            Cell indices to use. If None, uses all cells.
        batch_size
            Minibatch size for the DataLoader.
        library_size
            Library size normalization. If ``None``, uses each cell's observed
            library size. If a float, scales all cells to that total count. If
            ``"latent"``, returns log-rate-scale values (no library size
            multiplication).

        Returns
        -------
        np.ndarray
            Full expression of shape ``(N, G)``.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size or 128, shuffle=False
        )

        self.module.eval()
        results = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_idx = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
            species_idx = tensors[_SPECIES_REGISTRY_KEY].squeeze(-1).long()

            # Encode (deterministic: use z_mean, not reparameterized z)
            species_oh = F.one_hot(species_idx, self.module.n_species).float()
            batch_oh = F.one_hot(batch_idx, self.module.n_batch).float()
            enc_input = self.module._encoder_input(x, species_oh, batch_oh)
            h_enc = self.module.encoder(enc_input)
            z_mean = self.module.z_mean_linear(h_enc)

            # Decode (full path with residual)
            h = self.module.decoder_hidden(z_mean)
            log_rate_shared = self.module.output_shared(h)
            log_rate_residual = self.module.output_residual(
                torch.cat([h, species_oh, batch_oh], dim=-1)
            )

            # Batch offset
            constrained_offset = self.module.batch_offset()
            log_rate_batch = constrained_offset[batch_idx]

            # Alpha (per-gene integration coefficient)
            if hasattr(self.module, "_use_per_gene_alpha") and self.module._use_per_gene_alpha:
                alpha = torch.sigmoid(
                    torch.logit(self.module.conservation_prior)
                    + self.module.per_gene_correction
                )
            else:
                alpha, _ = self.module.gin(
                    self.module.gene_features, self.module.conservation_prior
                )

            # Combine: shared + batch offset + (1-alpha) * residual
            log_rate_full = (
                log_rate_shared
                + log_rate_batch
                + (1.0 - alpha) * log_rate_residual
            )

            if library_size == "latent":
                results.append(log_rate_full.cpu().numpy())
            else:
                if library_size is not None:
                    lib = float(library_size)
                else:
                    lib = x.sum(-1, keepdim=True)

                mu_full = torch.exp(log_rate_full) * lib
                results.append(mu_full.cpu().numpy())

        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Held-out evaluation (audit recommendation #3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_holdout(
        self,
        adata: AnnData | None = None,
        holdout_fraction: float = 0.2,
        seed: int = 42,
        n_neighbors: int = 15,
    ) -> dict:
        """Compute integration metrics on a held-out cell subset.

        Splits cells so that entire batches are held out (no within-batch
        leakage), then computes metrics on the held-out cells only.

        Parameters
        ----------
        adata
            AnnData to evaluate. If None, uses the registered AnnData.
        holdout_fraction
            Fraction of batches to hold out for evaluation.
        seed
            Random seed for batch selection.
        n_neighbors
            Number of neighbors for kNN-based metrics.

        Returns
        -------
        dict
            Dictionary with keys: ``sil_species``, ``sil_celltype``,
            ``label_transfer_acc``, ``n_cells_holdout``, ``holdout_batches``,
            ``z_var_per_dim``, ``z_n_dead_dims``.
        """
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import KNeighborsClassifier

        adata = self._validate_anndata(adata)

        # --- Batch-stratified holdout ---
        # Resolve the batch column: scvi-tools stores the encoded column in
        # the AnnDataManager registry, but the original column name is what
        # we need for interpretable batch IDs.
        batch_attr = None
        try:
            mgr = self._get_most_recent_anndata_manager(adata)
            batch_attr = mgr.data_registry["batch"]["attr_key"]
        except Exception:
            pass
        # Prefer the original "batch" column for interpretable names;
        # fall back to the scvi-encoded column.
        if "batch" in adata.obs.columns:
            batch_col = adata.obs["batch"]
        elif batch_attr and batch_attr in adata.obs.columns:
            batch_col = adata.obs[batch_attr]
        else:
            raise ValueError("Cannot find batch column in adata.obs")

        unique_batches = np.array(sorted(batch_col.unique()))
        rng = np.random.RandomState(seed)

        # If we have enough batches (>= 4), hold out entire batches ensuring
        # both species are represented. Otherwise, do a per-batch stratified
        # cell-level split so both species appear in the holdout.
        if len(unique_batches) >= 4:
            n_holdout = max(1, int(len(unique_batches) * holdout_fraction))
            holdout_batches = set(rng.choice(unique_batches, size=n_holdout, replace=False))
            holdout_mask = batch_col.isin(holdout_batches).values
        else:
            # Few batches: stratified cell-level split within each batch
            holdout_mask = np.zeros(adata.n_obs, dtype=bool)
            for b in unique_batches:
                b_idx = np.where(batch_col == b)[0]
                n_ho = max(1, int(len(b_idx) * holdout_fraction))
                chosen = rng.choice(b_idx, size=n_ho, replace=False)
                holdout_mask[chosen] = True
            holdout_batches = set(unique_batches)  # all batches represented

        train_mask = ~holdout_mask
        holdout_idx = np.where(holdout_mask)[0]
        train_idx = np.where(train_mask)[0]

        if len(holdout_idx) < 20:
            raise ValueError(
                f"Only {len(holdout_idx)} holdout cells. "
                f"Need more data for evaluation."
            )

        # --- Get latent representations ---
        z_all = self.get_latent_representation(adata=adata)
        z_holdout = z_all[holdout_idx]
        z_train = z_all[train_idx]

        # --- Species and cell-type labels ---
        # Prefer original "species" column over scvi-encoded "_scvi_labels"
        if "species" in adata.obs.columns:
            species_all = adata.obs["species"].values
        else:
            species_attr = None
            try:
                species_attr = mgr.data_registry["labels"]["attr_key"]
            except Exception:
                pass
            if species_attr and species_attr in adata.obs.columns:
                species_all = adata.obs[species_attr].values
            else:
                raise ValueError("Cannot find species column in adata.obs")
        species_holdout = species_all[holdout_idx]
        species_train = species_all[train_idx]

        # Cell type (look for common column names)
        ct_key = None
        for k in ["cell_type", "celltype", "cell_ontology_class", "CellType"]:
            if k in adata.obs.columns:
                ct_key = k
                break

        results = {
            "n_cells_holdout": len(holdout_idx),
            "n_cells_train": len(train_idx),
            "holdout_batches": list(holdout_batches),
        }

        # --- Silhouette scores on holdout ---
        if len(np.unique(species_holdout)) >= 2:
            results["sil_species"] = float(
                silhouette_score(z_holdout, species_holdout, metric="euclidean")
            )
        else:
            results["sil_species"] = float("nan")

        if ct_key and len(np.unique(adata.obs[ct_key].values[holdout_idx])) >= 2:
            ct_holdout = adata.obs[ct_key].values[holdout_idx]
            results["sil_celltype"] = float(
                silhouette_score(z_holdout, ct_holdout, metric="euclidean")
            )
        else:
            results["sil_celltype"] = float("nan")

        # --- Label transfer: train on species A holdout, predict species B holdout ---
        unique_species = np.unique(species_holdout)
        if len(unique_species) >= 2 and ct_key:
            ct_all = adata.obs[ct_key].values
            ct_holdout = ct_all[holdout_idx]
            # Shared cell types across species in holdout
            per_sp_types = [set(ct_holdout[species_holdout == s]) for s in unique_species]
            shared_types = per_sp_types[0]
            for s in per_sp_types[1:]:
                shared_types &= s

            if len(shared_types) >= 2:
                accs = []
                for train_sp in unique_species:
                    test_sps = [s for s in unique_species if s != train_sp]
                    tr_mask = (species_holdout == train_sp) & np.isin(ct_holdout, list(shared_types))
                    te_mask = np.isin(species_holdout, test_sps) & np.isin(ct_holdout, list(shared_types))
                    if tr_mask.sum() < 5 or te_mask.sum() < 5:
                        continue
                    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, tr_mask.sum()))
                    knn.fit(z_holdout[tr_mask], ct_holdout[tr_mask])
                    accs.append(float(knn.score(z_holdout[te_mask], ct_holdout[te_mask])))
                results["label_transfer_acc"] = float(np.mean(accs)) if accs else float("nan")
            else:
                results["label_transfer_acc"] = float("nan")
        else:
            results["label_transfer_acc"] = float("nan")

        # --- Per-dimension z variance on holdout (collapse check) ---
        z_var = np.var(z_holdout, axis=0)
        results["z_var_per_dim"] = z_var.tolist()
        results["z_n_dead_dims"] = int((z_var < 0.01).sum())
        results["z_n_active_dims"] = int((z_var >= 0.01).sum())
        results["z_mean_var"] = float(z_var.mean())

        return results

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return self._model_summary_string
