"""
Visualization Callback for I2fMRI Training
Generates plots during validation to monitor training progress
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Optional
import io
from PIL import Image


class VisualizationCallback(Callback):
    """
    Callback to visualize predictions during validation.

    Generates plots:
    1. Ground truth fMRI pattern (reconstructed to voxels if using parcellation)
    2. Predicted fMRI pattern (reconstructed to voxels if using parcellation)
    3. Correlation scatter plot (GT vs Pred)
    4. Per-voxel Pearson correlation heatmap
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 3,
        plot_types: List[str] = None,
        parcel_mapper = None,  # ParcelMapper for reconstruction
        save_dir: Optional[str] = None,  # Directory to save PNG files
    ):
        """
        Args:
            log_every_n_epochs: Log visualizations every N epochs
            num_samples: Number of samples to visualize
            plot_types: List of plot types to generate
            parcel_mapper: Optional ParcelMapper for reconstructing parcels to voxels
            save_dir: Directory to save visualization PNG files
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.plot_types = plot_types or [
            "ground_truth",
            "prediction",
            "correlation",
            "voxel_correlation"
        ]
        self.parcel_mapper = parcel_mapper
        self.save_dir = save_dir

        # Storage for validation outputs
        self.validation_outputs = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Store validation outputs for visualization."""
        # Only store first batch to save memory
        if batch_idx == 0:
            self.validation_outputs.append(outputs)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """Generate and log visualizations at the end of validation epoch."""
        # Check if we should log this epoch
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self.validation_outputs = []
            return

        if len(self.validation_outputs) == 0:
            return

        # Get first batch outputs
        outputs = self.validation_outputs[0]
        pred = outputs['pred'].detach().cpu().numpy()
        target = outputs['target'].detach().cpu().numpy()

        # Reconstruct from parcels to voxels if using parcellation
        if self.parcel_mapper is not None:
            print(f"Reconstructing {pred.shape[1]} parcels → {self.parcel_mapper.n_voxels} voxels for visualization")
            pred_voxels = self.parcel_mapper.reconstruct(pred)
            target_voxels = self.parcel_mapper.reconstruct(target)

            # Generate visualizations with reconstructed voxels
            figures = self._generate_plots(pred_voxels, target_voxels, trainer.current_epoch,
                                          is_reconstructed=True)
        else:
            # Generate visualizations with original data
            figures = self._generate_plots(pred, target, trainer.current_epoch,
                                          is_reconstructed=False)

        # Save to PNG files if save_dir is provided
        if self.save_dir is not None:
            import os
            epoch_dir = os.path.join(self.save_dir, f'epoch_{trainer.current_epoch:04d}')
            os.makedirs(epoch_dir, exist_ok=True)

            for name, fig in figures.items():
                save_path = os.path.join(epoch_dir, f'{name}.png')
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization: {save_path}")

        # Log to tensorboard or wandb
        if trainer.logger is not None:
            for name, fig in figures.items():
                # Convert figure to image
                img = self._fig_to_image(fig)

                # Log based on logger type
                if hasattr(trainer.logger, 'experiment'):
                    if hasattr(trainer.logger.experiment, 'add_image'):
                        # TensorBoard
                        trainer.logger.experiment.add_image(
                            f'visualizations/{name}',
                            img,
                            trainer.current_epoch,
                            dataformats='HWC'
                        )
                    elif hasattr(trainer.logger.experiment, 'log'):
                        # WandB
                        import wandb
                        trainer.logger.experiment.log({
                            f'visualizations/{name}': wandb.Image(img)
                        })

                plt.close(fig)

        # Clear outputs
        self.validation_outputs = []

    def _generate_plots(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        epoch: int,
        is_reconstructed: bool = False
    ) -> Dict[str, plt.Figure]:
        """
        Generate all requested plots.

        Args:
            pred: Predicted fMRI [B, N]
            target: Ground truth fMRI [B, N]
            epoch: Current epoch number
            is_reconstructed: Whether data was reconstructed from parcels

        Returns:
            Dictionary of figure name -> matplotlib Figure
        """
        figures = {}
        num_samples = min(self.num_samples, pred.shape[0])

        # Add title suffix if reconstructed
        title_suffix = " (Reconstructed from 1000 Parcels)" if is_reconstructed else ""

        # 1. Ground truth and prediction plots (side by side for multiple samples)
        if "ground_truth" in self.plot_types or "prediction" in self.plot_types:
            fig, axes = plt.subplots(
                num_samples, 2,
                figsize=(12, 4 * num_samples)
            )
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                # Ground truth
                axes[i, 0].plot(target[i], linewidth=0.5, alpha=0.7)
                axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
                axes[i, 0].set_xlabel('Voxel Index' if is_reconstructed else 'Voxel/Parcel Index')
                axes[i, 0].set_ylabel('fMRI Signal')
                axes[i, 0].grid(True, alpha=0.3)

                # Prediction
                axes[i, 1].plot(pred[i], linewidth=0.5, alpha=0.7, color='orange')
                axes[i, 1].set_title(f'Sample {i+1}: Prediction')
                axes[i, 1].set_xlabel('Voxel Index' if is_reconstructed else 'Voxel/Parcel Index')
                axes[i, 1].set_ylabel('fMRI Signal')
                axes[i, 1].grid(True, alpha=0.3)

                # Compute and display Pearson correlation
                corr, _ = pearsonr(pred[i], target[i])
                axes[i, 1].text(
                    0.02, 0.98,
                    f'Pearson: {corr:.4f}',
                    transform=axes[i, 1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            plt.suptitle(f'Epoch {epoch}: Ground Truth vs Prediction{title_suffix}', fontsize=14)
            plt.tight_layout()
            figures['gt_vs_pred'] = fig

        # 2. Correlation scatter plots
        if "correlation" in self.plot_types:
            fig, axes = plt.subplots(
                1, num_samples,
                figsize=(5 * num_samples, 4)
            )
            if num_samples == 1:
                axes = [axes]

            for i in range(num_samples):
                # Scatter plot
                axes[i].scatter(
                    target[i], pred[i],
                    alpha=0.3, s=1, c='blue'
                )

                # Diagonal line (perfect prediction)
                min_val = min(target[i].min(), pred[i].min())
                max_val = max(target[i].max(), pred[i].max())
                axes[i].plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    'r--', linewidth=2, label='Perfect Prediction'
                )

                # Compute metrics
                corr, _ = pearsonr(pred[i], target[i])
                mse = np.mean((pred[i] - target[i]) ** 2)

                axes[i].set_xlabel('Ground Truth')
                axes[i].set_ylabel('Prediction')
                axes[i].set_title(f'Sample {i+1}\nPearson: {corr:.4f}, MSE: {mse:.4f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

            plt.suptitle(f'Epoch {epoch}: Correlation Plots{title_suffix}', fontsize=14)
            plt.tight_layout()
            figures['correlation_scatter'] = fig

        # 3. Per-voxel Pearson correlation (averaged across samples)
        if "voxel_correlation" in self.plot_types:
            num_voxels = pred.shape[1]

            # Compute per-voxel correlation across all samples
            voxel_corrs = []
            for v in range(num_voxels):
                pred_v = pred[:, v]
                target_v = target[:, v]

                # Check if there's variation
                if pred_v.std() > 0 and target_v.std() > 0:
                    corr, _ = pearsonr(pred_v, target_v)
                    voxel_corrs.append(corr)
                else:
                    voxel_corrs.append(0.0)

            voxel_corrs = np.array(voxel_corrs)

            # Create heatmap
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Line plot
            axes[0].plot(voxel_corrs, linewidth=0.5, alpha=0.7)
            axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0].set_xlabel('Voxel Index' if is_reconstructed else 'Voxel/Parcel Index')
            axes[0].set_ylabel('Pearson Correlation')
            axes[0].set_title(f'Per-Voxel Pearson Correlation (Mean: {voxel_corrs.mean():.4f})')
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Histogram
            axes[1].hist(voxel_corrs, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=voxel_corrs.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {voxel_corrs.mean():.4f}')
            axes[1].set_xlabel('Pearson Correlation')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Distribution of Per-Voxel Correlations')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(f'Epoch {epoch}: Per-Voxel Analysis{title_suffix}', fontsize=14)
            plt.tight_layout()
            figures['voxel_correlation'] = fig

        return figures

    @staticmethod
    def _fig_to_image(fig: plt.Figure) -> np.ndarray:
        """
        Convert matplotlib figure to numpy array.

        Args:
            fig: Matplotlib figure

        Returns:
            Image as numpy array [H, W, C]
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        buf.close()
        return img_array
