import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import evaluation_metrics.evaluation_metrics as eval

from trainfree import TrainFree
from itertools import combinations
from diffseg.diffseg.segmentor import DiffSeg
from diffseg.diffseg.utils import process_image, augmenter
from diffseg.third_party.keras_cv.stable_diffusion import StableDiffusion
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from base.generative_semantic_segmentator import GenerativeSemanticSegmentator


class DiffSeg(GenerativeSemanticSegmentator):
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
        # KL_THRESHOLD controls the merging threshold
        KL_THRESHOLD = [0.9]*3
        NUM_POINTS = 16
        REFINEMENT = True

        with tf.device('/GPU:0'):
            image_encoder = ImageEncoder()
            vae = tf.keras.Model(
                image_encoder.input,
                image_encoder.layers[-1].output,
            )
            model = StableDiffusion(img_width=512, img_height=512)
            latent = vae(
                tf.expand_dims(
                    augmenter(process_image(self.image)),
                    axis=0
                ), training=False
            )
            _, weight_64, weight_32, weight_16, weight_8, _, _, _, _ = model.text_to_image(
                None,
                batch_size=1,
                latent=latent,
                timestep=300
            )
            segmentor = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)
            predictions = segmentor.segment(
                weight_64, weight_32, weight_16, weight_8)
            self.predictions = predictions

    def postprocess(self):
        # ToDo: Manage load of predictions (no hooker)

        list_semantic_masks = [
            self._postprocess_best_match(),
            self._postprocess_attention_match()
        ]

        self.list_semantic_masks = list_semantic_masks

        return list_semantic_masks

    def plot(self):
        initial_images = 3
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        for ax in axes:
            ax.axis("off")
            ax.imshow(self.image)

        axes[0].set_title("Synthetized image")

        axes[1].set_title("Semantic mask with threshold 0.85")
        axes[1].imshow(
            self.greedy_semantic_mask,
            alpha=self.greedy_semantic_mask,
            cmap='jet'
        )

        x, y = self._find_edges(self.predictions)
        axes[2].set_title("Overlay proposals")
        axes[2].imshow(
            self.predictions[0],
            cmap='jet',
            alpha=0.5,
            vmin=-1,
            vmax=len(set(self.predictions.flatten()))
        )
        axes[2].scatter(x, y, color="blue", s=0.5)

        axes[3].set_title("Attention match semantic mask")
        axes[3].imshow(
            self.attention_match,
            alpha=self.attention_match,
            cmap='jet'
        )

        axes[4].set_title("Best match semantic mask")
        axes[4].imshow(
            self.best_match,
            alpha=self.best_match,
            cmap='jet'
        )

        fig.tight_layout()

    def _postprocess_best_match(self):
        best_match = None
        max_iou = 0

        all_combinations = self._generate_all_combinations(
            len(set(self.predictions.flatten()))
        )
        for comb in all_combinations:
            prediction_copy = self.predictions.copy()
            mask = np.isin(prediction_copy[0], comb)
            prediction_copy[0][~mask] = 0
            prediction_copy[0][mask] = 1

            current_iou = round(
                eval.compute_intersection_over_union(
                    self.annotation, prediction_copy[0]
                ), 3
            )
            if current_iou > max_iou:
                max_iou = current_iou
                best_match = prediction_copy[0].astype(float)

            self.best_match = best_match

            return best_match

    def _generate_all_combinations(self, max_number_list):
        elements = list(range(max_number_list+1))

        all_combinations = []
        for r in range(len(elements) + 1):
            comb = list(combinations(elements, r))
            all_combinations.extend(comb)

        all_combinations = [list(comb)
                            for comb in all_combinations if len(comb) != 0]
        return all_combinations

    def _postprocess_attention_match(self):
        attention_match = None

        greedy_semantic_mask = None
        train_free_method = TrainFree(
            image=self.image,
            hooker=self.hooker,
            prompt=self.prompt,
            list_thresholds=[0.85],
            list_thresholds_str=None,
            token_idx=self.token_idx,
            annotation=None,
        )
        train_free_method.combine()
        greedy_semantic_mask = train_free_method.postprocess()[0]
        self.greedy_semantic_mask = greedy_semantic_mask

        matches_found = []
        for idx in range(len(set(self.predictions.flatten()))):
            prediction_copy = self.predictions[0].astype(float).copy()
            mask = np.isin(prediction_copy, [idx])
            prediction_copy[~mask] = 0
            prediction_copy[mask] = 1

            intersection = np.logical_and(
                greedy_semantic_mask, prediction_copy)
            matches = np.any(np.any(intersection == 1, axis=1))

            if matches:
                matches_found.append(idx)

        attention_match = self.predictions[0].astype(float).copy()
        mask = np.isin(attention_match, matches_found)
        attention_match[~mask] = 0
        attention_match[mask] = 1
        self.attention_match = attention_match

        return attention_match

    def _find_edges(self, M):
        edges = np.zeros((512, 512))
        m1 = M[1:510, 1:510] != M[0:509, 1:510]
        m2 = M[1:510, 1:510] != M[2:511, 1:510]
        m3 = M[1:510, 1:510] != M[1:510, 0:509]
        m4 = M[1:510, 1:510] != M[1:510, 2:511]
        edges[1:510, 1:510] = (m1 | m2 | m3 | m4).astype(int)
        x_new = np.linspace(0, 511, 512)
        y_new = np.linspace(0, 511, 512)
        x_new, y_new = np.meshgrid(x_new, y_new)
        x_new = x_new[edges == 1]
        y_new = y_new[edges == 1]
        return x_new, y_new
