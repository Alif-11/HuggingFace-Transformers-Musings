import transformers
from typing import List

def zero_shot_classification_tryout(sentence : str, list_of_labels : List[str]):
  zero_shot_classifier = transformers.pipeline("zero-shot-classification")
  label_scores = zero_shot_classifier(sentence, candidate_labels=list_of_labels)
  return label_scores

# Zero Shot Classification Tryout
# print(zero_shot_classification_tryout("Why is the sky blue? Does that have any impact on global governance? What if the sky was red? Are starfish a type of crustacean or echinoderm?",["science","governance","policy","finance","unga","food"]))