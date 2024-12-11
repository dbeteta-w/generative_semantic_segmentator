import torch

import numpy as np
import matplotlib.pyplot as plt

from base.generative_semantic_segmentator import GenerativeSemanticSegmentator


class TrainFree(GenerativeSemanticSegmentator):
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
        ovam_evaluator = self.hooker.get_ovam_callable(
            expand_size=(512, 512),
            heads_epochs_aggregation=self._extract_and_weight_by_dimension
        )
        with torch.no_grad():
            attention_maps = ovam_evaluator(self.prompt)
            attention_maps = attention_maps[0].cpu().numpy()

        if self.filename:
            np.save(self.filename[0], attention_maps)

        self.attention_maps = attention_maps

        self_attention_maps = [
            self_attention_map.cpu()
            for self_attention_map in self.hooker.get_self_attention_map(stack=False)
        ]
        if self.filename:
            np.save(self.filename[1], self_attention_maps)

        self.self_attention_maps = self_attention_maps

    def postprocess(self):
        list_semantic_maks = []

        if not self.attention_maps or len(self.attention_maps) == 0:
            self.attention_maps = np.load(self.filename[0])
            self.self_attention_maps = np.load(self.filename[1])

        # Filter the self-attention maps with 64 dimension
        normalized_self_attention_maps = self._process_self_attention_maps(
            self.self_attention_maps
        )
        (
            attention_map_token,
            normalized_attention_map_token
        ) = self._select_token_and_normalize(self.attention_maps)

        self.attention_map_token = attention_map_token
        self.normalized_attention_map_token = normalized_attention_map_token

        # Fuse attention_maps with the info of the self_attention_maps
        fused_attention_map = normalized_self_attention_maps * \
            normalized_attention_map_token

        # Min-max scalation
        minimun = 0
        maximun = 1
        self_attention_minimun = fused_attention_map.min()
        self_attention_maximun = fused_attention_map.max()
        scaled_attention_map = (fused_attention_map - self_attention_minimun) / \
            (self_attention_maximun - self_attention_minimun)
        scaled_attention_map = scaled_attention_map * \
            (maximun - minimun) + minimun

        for threshold in self.list_thresholds:
            semantic_mask = (scaled_attention_map > threshold).astype(float)
            list_semantic_maks.append(semantic_mask)

        self.list_semantic_maks = list_semantic_maks

        return list_semantic_maks

    def plot(self):
        initial_images = 2
        fig, axes = plt.subplots(
            1, len(self.list_thresholds) + initial_images,
            figsize=(20, 5)
        )

        for ax in axes:
            ax.axis("off")
            ax.imshow(self.image)

        axes[0].set_title("Synthetized image")

        axes[1].set_title("Soft semantic mask")
        axes[1].imshow(
            self.attention_map_token,
            alpha=self.normalized_attention_map_token,
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

    def _extract_and_weight_by_dimension(self, attention_maps):
        """
        Input: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        Output: (heads, n_tokens, latent_size / factor, latent_size / factor)

        Dimension 8 does not exist, we need to normalize:
        Initial: (0.3; 0.5; 0.1; 0.1) Final: (0.5 / 0.7; 0.1 / 0.7; 0.1 / 0.7)
        """
        ponderation_dict = {
            16: 0.714,
            32: 0.143,
            64: 0.143,
        }
        dimension = attention_maps.shape[3]
        heatmap_ponderated = attention_maps * ponderation_dict[dimension]
        return torch.sum(heatmap_ponderated, dim=0)

    def _process_self_attention_maps(self, self_attention_maps):
        dimension_to_filter = 64

        for idx in range(len(self_attention_maps)):
            if self_attention_maps[idx].shape[0] == dimension_to_filter:
                interpolated_self_attention_map = torch.nn.functional.interpolate(
                    self_attention_maps[idx].unsqueeze(0).unsqueeze(0),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                self_attention_maps[idx] = interpolated_self_attention_map

        filtered_self_attention_maps = [
            self_attention_map for self_attention_map in self_attention_maps
            if self_attention_map.shape[0] == 512
        ]
        mean_self_attention_map = torch.stack(
            filtered_self_attention_maps
        ).mean(axis=0).cpu().numpy()
        normalized_self_attention_maps = (
            mean_self_attention_map / mean_self_attention_map.max()
        )

        return normalized_self_attention_maps
