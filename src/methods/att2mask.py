import torch

import numpy as np
import matplotlib.pyplot as plt

from base.generative_semantic_segmentator import GenerativeSemanticSegmentator


class Att2Mask(GenerativeSemanticSegmentator):
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
        # Extract one timestep
        ovam_evaluator = self.hooker.get_ovam_callable(
            expand_size=(512, 512),
            heads_epochs_aggregation=self._extract_one_tp
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

        for beta, dcrf in self.list_thresholds:
            background_attention = 1 - normalized_attention_map_token - beta
            token_and_background = np.stack(
                [background_attention, normalized_attention_map_token], axis=0
            )
            # Choose pixel with max value from background or token
            semantic_mask = token_and_background.argmax(
                axis=0).astype(np.float32)
            if dcrf:
                semantic_mask = self._densecrf(
                    np.array(self.image), semantic_mask)

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

        axes[1].set_title("Soft semantic mask of one time-step")
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

    def _extract_one_tp(self, heatmap):
        """
        Input: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        Output: (heads, n_tokens, latent_size / factor, latent_size / factor)
        """

        # Instead of choosing the 50th of 100th timesteps, get the equivalent: total_amount_time_steps // 2
        timestep = heatmap.shape[0] // 2
        return heatmap[timestep, ...]

    def _densecrf(self, image, semantic_mask, w1=10.0, alpha=80, beta=13, w2=3.0, gamma=3, it=5.0):
        """
        Input:
            I    : a numpy array of shape [H, W, C], where C should be 3.
                  type of I should be np.uint8, and the values are in [0, 255]
            P    : a probability map of shape [H, W, L], where L is the number of classes
                  type of P should be np.float32
            param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                    w1    :   weight of bilateral term, e.g. 10.0
                    alpha :   spatial distance std, e.g., 80
                    beta  :   rgb value std, e.g., 15
                    w2    :   weight of spatial term, e.g., 3.0
                    gamma :   spatial distance std for spatial term, e.g., 3
                    it    :   iteration number, e.g., 5
        Output:
            out  : a numpy array of shape [H, W], where pixel values represent class indices.
        """
        image = np.array(image, dtype=np.uint8)
        semantic_mask = np.array(semantic_mask, dtype=np.float32)
        if len(semantic_mask.shape) == 2:
            semantic_mask = np.stack(
                [1 - semantic_mask, semantic_mask], axis=-1)
        param = (w1, alpha, beta, w2, gamma, it)
        out = denseCRF.densecrf(image, semantic_mask, param)
        return out
