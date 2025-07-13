import transformers
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

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
# print(text_generation("Todos de los hombres del rey lo", 2, 15))

def sentiment_analysis_comparison(list_of_starting_texts : List[str]):
  sentiment_analysis_classifer = transformers.pipeline("sentiment-analysis")

  desired_model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
  tokenizer = AutoTokenizer.from_pretrained(desired_model_checkpoint)
  huggingface_tokenized_inputs = tokenizer(list_of_starting_texts, padding=True, truncation=True, return_tensors="pt")

  sentiment_analysis_classifer_auto_model = AutoModelForSequenceClassification.from_pretrained(desired_model_checkpoint)
  auto_model_output = sentiment_analysis_classifer_auto_model(**huggingface_tokenized_inputs)
  auto_model_predictions = torch.nn.functional.softmax(auto_model_output.logits, dim=-1)
  #print(sentiment_analysis_classifer_auto_model.config.id2label)
  #print(auto_model_predictions)

  argmax_per_prediction_scores = np.argmax(auto_model_predictions.detach().numpy(),axis=-1)
  max_per_prediction_scores = np.max(auto_model_predictions.detach().numpy(), axis=-1)
  #print(argmax_per_prediction_scores)
  #print(max_per_prediction_scores)

  pipeline_predictions = dict()
  argmax_counter = 0

  print(argmax_per_prediction_scores)

  while argmax_counter < len(argmax_per_prediction_scores):
    argmax = argmax_per_prediction_scores[argmax_counter]
    maximum = max_per_prediction_scores[argmax_counter]
    pipeline_predictions['label'] = sentiment_analysis_classifer_auto_model.config.id2label[argmax]
    pipeline_predictions['score'] = maximum
    argmax_counter += 1
  return sentiment_analysis_classifer(list_of_starting_texts), pipeline_predictions


# Sentiment Analysis Comparison Tryout
# See behavior of sentiment analysis pipeline
# As well as the behavior of using Tokenizer, Model individually
#print(sentiment_analysis_comparison(["I think you're really cool. But I hate the taste of broccoli!", "I... want to make things riiiiiii-ight! Wanna make them the way they're supposed to be!"]))

