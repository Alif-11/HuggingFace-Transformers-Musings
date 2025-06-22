import transformers
from typing import List

def zero_shot_classification_tryout(sentence : str, list_of_labels : List[str]):
  zero_shot_classifier = transformers.pipeline("zero-shot-classification")
  label_scores = zero_shot_classifier(sentence, candidate_labels=list_of_labels)
  return label_scores

# Zero Shot Classification Tryout
# print(zero_shot_classification_tryout("Why is the sky blue? Does that have any impact on global governance? What if the sky was red? Are starfish a type of crustacean or echinoderm?",["science","governance","policy","finance","unga","food"]))

def text_generation(starting_sentence_fragment : str, num_return_sequences: int, max_length: int):
  text_generator = transformers.pipeline("text-generation")
  return text_generator(starting_sentence_fragment, num_return_sequences=num_return_sequences, max_length=max_length, max_new_tokens=None)


# Text Generation Tryout
print(text_generation("Todos de los hombres del rey lo", 2, 15))