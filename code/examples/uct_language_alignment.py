import transformers
from transformers import pipeline, DistilBertModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from dyna_gym.pipelines import uct_for_hf_transformer_pipeline
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# model set up:
model_name = "gpt2"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

vocab_size = tokenizer.vocab_size
horizon = 50  # maximum number of steps / tokens to generate in each episode

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def sentiment_analysis(sentence):
    """
    define a reward function based on sentiment of the generated text
    """
    output = sentiment_pipeline(sentence)[0]
    if output['label'] == 'POSITIVE':
        return output['score']
    else:
        return -output['score']


# arguments for the UCT agent
uct_args = dict(
    rollouts=20,
    gamma=1.,
    width=3,
    alg='uct',  # or p_uct
)

# will be passed to huggingface model.generate()
model_generation_args = dict(
    top_k=3,
    top_p=0.9,
    do_sample=True,
    temperature=0.7,
)

pipeline = uct_for_hf_transformer_pipeline(
    model=model,
    tokenizer=tokenizer,
    horizon=horizon,
    reward_func=sentiment_analysis,
    uct_args=uct_args,
    model_generation_args=model_generation_args,
    should_plot_tree=True,  # plot the tree after generation
)

input_str = "What do you think of this movie?"
outputs = pipeline(input_str=input_str)

for text, reward in zip(outputs['texts'], outputs['rewards']):
    print("==== Text ====")
    print(text)
    print("==== Reward:", reward, "====")
    print()

"""

def main():
    sentence = "This is wrong."
    return sentiment_analysis(sentence)


if __name__ == "__main__":
    #Test the reward function
    #sentence = "This is wrong."
    #reward = sentiment_analysis(sentence)
    #print(f"Sentence: {sentence}")
    #print(f"Sentiment-based reward: {reward}")
    print(main())

"""
