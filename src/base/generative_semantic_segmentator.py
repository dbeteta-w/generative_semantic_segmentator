from abc import ABC, abstractmethod
import evaluation_metrics.evaluation_metrics as eval 

class GenerativeSemanticSegmentator(ABC):
    attention_maps = None
    attention_map_token = None
    normalized_attention_map_token = None
    list_semantic_maks = None

    def __init__(
          self, image, hooker, prompt,
          list_thresholds, list_thresholds_str,
          token_idx, annotation, filename
        ):
        self.image = image
        self.hooker = hooker
        self.prompt = prompt
        self.list_thresholds = list_thresholds
        self.list_thresholds_str = list_thresholds_str
        self.token_idx = token_idx
        self.annotation = annotation
        self.filename = filename

    @abstractmethod
    def combine(self):
        """Combine attention maps following a given method"""
        pass


    @abstractmethod
    def postprocess(self):
        """Postprocess attention maps following a given method"""
        pass

    @abstractmethod
    def plot(self):
        """Plot segmentation maps"""
        pass


    def evaluate(self):
      """Evaluate segmentation maps: IoU & BA"""

      list_iou = []
      list_ba = []

      for semantic_mask in self.list_semantic_maks:
          iou = eval.compute_intersection_over_union(
              self.annotation, semantic_mask
            )
          list_iou.append(iou)

          ba = eval.compute_balanced_accuracy(
              self.annotation, semantic_mask
            )
          list_ba.append(ba)

      self.list_iou = list_iou
      self.list_ba = list_ba

      return (list_iou, list_ba)


    def _select_token_and_normalize(self, attention_maps):
      """Select token idx and normalize the attention map"""

      attention_map_token = attention_maps[self.token_idx]
      normalized_attention_map_token = (
          attention_map_token / attention_map_token.max()
      )
      return (attention_map_token, normalized_attention_map_token)