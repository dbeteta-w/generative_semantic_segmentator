import torch

import numpy as np
import matplotlib.pyplot as plt

from base.generative_semantic_segmentator import GenerativeSemanticSegmentator


class DAAM(GenerativeSemanticSegmentator):
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
            expand_size=(512, 512)
        )
        with torch.no_grad():
            attention_maps = ovam_evaluator(self.prompt)
            attention_maps = attention_maps[0].cpu().numpy()

        if self.filename:
            np.save(self.filename, attention_maps)

        self.attention_maps = attention_maps

    def postprocess(self):
        list_semantic_maks = []

        if not self.attention_maps or len(self.attention_maps) == 0:
            self.attention_maps = np.load(self.filename)

        (
            attention_map_token,
            normalized_attention_map_token
        ) = self._select_token_and_normalize(self.attention_maps)

        self.attention_map_token = attention_map_token
        self.normalized_attention_map_token = normalized_attention_map_token

        for threshold in self.list_thresholds:
            semantic_mask = (normalized_attention_map_token >
                             threshold).astype(float)
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
