from collections import OrderedDict

import gymnasium as gym
import torch
import transformers
from transformers import pipeline
from gymnasium.spaces import Discrete,Box
import  numpy as np


class LanguageEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """

    def __init__(self, terminal_token, horizon, reward_func, vocab_size):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon
        self.get_reward = reward_func

        # this line from me;
        self.vocab_size = vocab_size
        self.action_space = Discrete(vocab_size)
        self.observation_space = Box(low=0, high=vocab_size, shape=(horizon,))

    def reset(self, input_ids, attention_mask=None):
        # ini or reset the environment to give a starting state
        if attention_mask is not None:
            attention_mask = attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)

        self.state = (input_ids, attention_mask)
        self.input_len = len(input_ids)
        return self.state

    def transition(self, s, a, is_model_dynamic=False):
        """
        s: current state, which is a tuple (ids, attention_mask)
        a: action, a token ID
        is_model_dynamic: placeholder
        """

        ids, attention_mask = s

        # s is a one-dimensional tensor, a is a token id (scalar), concatenate them to form a new state
        next_ids = torch.cat([ids, torch.tensor([a]).to(ids.device)])
        # append a 1 to the attention mask
        attention_mask = torch.cat([attention_mask, torch.tensor([1]).to(attention_mask.device)])

        if a == self.terminal_token or len(next_ids) == self.horizon:
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward((next_ids, attention_mask))
        else:
            reward = 0  # no intermediate reward

        return (next_ids, attention_mask), reward, done

    def step(self, action):
        # take a step in the environment using the given action
        self.state, reward, done = self.transition(self.state, action)
        return self.state, reward, done, {}

    def equality_operator(self, s1, s2):
        # s1 and s2 are two tensors
        # compare two states to check if they are identical
        return all(torch.equal(x1, x2) for x1, x2 in zip(s1, s2))


def main():
    # Define terminal token and horizon
    model_name = "gpt2"
    terminal_token = 50256  # GPT-2's EOS token
    horizon = 50

    # Initialize GPT-2 model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    # Define a reward function based on sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis")

    def sentiment_analysis_reward(state):
        ids, _ = state
        text = tokenizer.decode(ids, skip_special_tokens=True)
        output = sentiment_pipeline(text)[0]
        return output['score'] if output['label'] == 'POSITIVE' else -output['score']

    # Initialize the environment
    env = LanguageEnv(terminal_token, horizon, sentiment_analysis_reward, vocab_size)

    # Reset the environment
    input_ids = tokenizer.encode("What do you think of this movie?", return_tensors="pt")[0]
    attention_mask = torch.ones_like(input_ids)
    state = env.reset(input_ids, attention_mask)

    print("Initial state:", tokenizer.decode(state[0], skip_special_tokens=True))

    # Take actions in the environment
    done = False
    while not done:
        logits = model(input_ids.unsqueeze(0)).logits[0, -1, :]
        next_token = torch.argmax(logits).item()
        state, reward, done, _ = env.step(next_token)
        input_ids = state[0]
        print(
            f"Next token: {next_token}, State: {tokenizer.decode(input_ids, skip_special_tokens=True)}, Reward: {reward}, Done: {done}")

    print("Final state:", tokenizer.decode(input_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
