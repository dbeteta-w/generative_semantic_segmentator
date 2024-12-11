import math
import torch
import numbers

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from base.generative_semantic_segmentator import GenerativeSemanticSegmentator


class AttExc(GenerativeSemanticSegmentator):
    def __init__(
        self, image, hooker, prompt,
        list_thresholds, list_thresholds_str,
        token_idx, annotation, filename
    ):
        super().__init__(
            image, hooker, prompt,
            list_thresholds, list_thresholds_str,
            token_idx, annotation, filename
        )

    def combine(self):
       # Extract one dimension from the first half of timesteps
        ovam_evaluator = self.hooker.get_ovam_callable(
            expand_size=(512, 512),
            heads_epochs_aggregation=self._extract_one_dim_half_tp
        )
        with torch.no_grad():
            attention_maps = ovam_evaluator(self.prompt)
            attention_maps_one_dim_half_tp = attention_maps[0].cpu().numpy()

        if self.filename:
            np.save(self.filename[0], attention_maps_one_dim_half_tp)

        # Extract one dimension from all the timesteps
        ovam_evaluator = self.hooker.get_ovam_callable(
            expand_size=(512, 512),
            heads_epochs_aggregation=self._extract_one_dim_all_tp
        )
        with torch.no_grad():
            attention_maps = ovam_evaluator(self.prompt)
            attention_maps_one_dim_all_tp = attention_maps[0].cpu().numpy()

        if self.filename:
            np.save(self.filename[1], attention_maps_one_dim_all_tp)

        self.attention_maps = (
            attention_maps_one_dim_half_tp,
            attention_maps_one_dim_all_tp
        )

    def postprocess(self):
        list_semantic_maks = []

        if not self.attention_maps or len(self.attention_maps) == 0:
            attention_maps_one_dim_half_tp = np.load(self.filename[0])
            attention_maps_one_dim_all_tp = np.load(self.filename[1])
            self.attention_maps = (
                attention_maps_one_dim_half_tp,
                attention_maps_one_dim_all_tp
            )

        attention_maps_one_dim_half_tp, attention_maps_one_dim_all_tp = self.attention_maps
        (
            attention_map_token_one_dim_half_tp,
            normalized_attention_map_token_one_dim_half_tp
        ) = self._select_token_and_normalize(attention_maps_one_dim_half_tp)
        (
            attention_map_token_one_dim_all_tp,
            normalized_attention_map_token_one_dim_all_tp
        ) = self._select_token_and_normalize(attention_maps_one_dim_all_tp)

        self.attention_map_token = (
            attention_map_token_one_dim_half_tp,
            attention_map_token_one_dim_all_tp,
        )
        self.normalized_attention_map_token = (
            normalized_attention_map_token_one_dim_half_tp,
            normalized_attention_map_token_one_dim_all_tp
        )

        self.token_idx -= 1
        smooth_attention_maps_one_dim_half_tp = self._smooth_attention_maps(
            attention_maps_one_dim_half_tp
        )
        (
            _, normalized_attention_map_token_one_dim_half_tp
        ) = self._select_token_and_normalize(
            smooth_attention_maps_one_dim_half_tp
        )

        smooth_attention_maps_one_dim_all_tp = self._smooth_attention_maps(
            attention_maps_one_dim_all_tp
        )
        (
            _, normalized_attention_map_token_one_dim_all_tp
        ) = self._select_token_and_normalize(
            smooth_attention_maps_one_dim_all_tp
        )

        for first_threshold, second_threshold in self.list_thresholds:
            semantic_map_one_dim_half_tp = (
                (normalized_attention_map_token_one_dim_half_tp >
                 first_threshold).astype(float)
            )
            semantic_maps_one_dim_all_tp = (
                (normalized_attention_map_token_one_dim_all_tp >
                 second_threshold).astype(float)
            )

            list_semantic_maks.append(
                np.logical_or(
                    semantic_map_one_dim_half_tp, semantic_maps_one_dim_all_tp
                ).astype(float)
            )

        self.list_semantic_maks = list_semantic_maks

        return list_semantic_maks

    def plot(self):
        initial_images = 3
        fig, axes = plt.subplots(
            1, len(self.list_thresholds) + initial_images,
            figsize=(20, 5)
        )

        for ax in axes:
            ax.axis("off")
            ax.imshow(self.image)

        axes[0].set_title("Synthetized image")

        axes[1].set_title("Soft semantic mask of one dim & first half of tp")
        axes[1].imshow(
            self.attention_map_token[0],
            alpha=self.normalized_attention_map_token[0],
            cmap='jet'
        )

        axes[2].set_title("Soft semantic mask of one dime & all the tp")
        axes[2].imshow(
            self.attention_map_token[1],
            alpha=self.normalized_attention_map_token[1],
            cmap='jet'
        )

        for idx in range(len(self.list_thresholds)):
            axes[idx + initial_images].set_title(
                f"Semantic mask with: {self.list_thresholds_str[idx]}"
            )
            axes[idx + initial_images].imshow(
                self.list_semantic_maks[idx],
                alpha=self.list_semantic_maks[idx],
                cmap='jet'
            )

        fig.tight_layout()

    def _extract_one_dim_half_tp(self, attention_maps):
        """
        Input: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        Output: (heads, n_tokens, latent_size / factor, latent_size / factor)
        """

        dimension = attention_maps.shape[3]
        timestep = attention_maps.shape[0] // 2

        if dimension == 16:
            return torch.sum(attention_maps[:timestep, ...], dim=0)
        else:
            # shape = (heads, n_tokens, dimension, dimension)
            return torch.sum(attention_maps * 0, dim=0)

    def _extract_one_dim_all_tp(self, attention_maps):
        """
        Input: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        Output: (heads, n_tokens, latent_size / factor, latent_size / factor)
        """

        dimension = attention_maps.shape[3]

        if dimension == 16:
            return torch.sum(attention_maps, dim=0)
        else:
            # shape = (heads, n_tokens, dimension, dimension)
            return torch.sum(attention_maps * 0, dim=0)

    def _smooth_attention_maps(self, attention_maps):
        # (8-1, 512, 512) Discard special token (<SOT>)
        attention_maps = attention_maps[1:, :, :]

        # Softmax
        attention_maps = self._softmax(attention_maps)
        # Convert numpy array to PyTorch tensor
        attention_maps = torch.from_numpy(attention_maps).float()

        # Gaussian filter per token
        smooth_attention_maps = []
        smooth = GaussianSmoothing(
            channels=1, kernel_size=3, sigma=0.5, dim=2
        )
        for idx in range(attention_maps.shape[0]):
            # Expand dims to match the expected input shape for conv2d
            input_map = attention_maps[idx, :, :].unsqueeze(0).unsqueeze(0)
            input_map = F.pad(input_map, (1, 1, 1, 1), mode='reflect')
            smooth_attention_map = smooth(input_map).squeeze(0).squeeze(0)
            smooth_attention_maps.append(smooth_attention_map)

        # Convert the result back to numpy array
        smooth_attention_maps = torch.stack(
            smooth_attention_maps).cpu().numpy()

        return smooth_attention_maps

    def _softmax(self, x, axis=0):
        # Subtract the max value for numerical stability
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ], indexing='ij'  # Adding the indexing argument as required
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
