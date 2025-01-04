import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import pandas as pd
import subprocess
import shlex
import zipfile
import torch
from datasets import load_dataset
import os
import numpy as np
import os
import json
import os
import ast
from concurrent.futures import ThreadPoolExecutor

# Assume openai>=1.0.0
from openai import OpenAI

DEEPINFRA_TOKEN=os.getenv('DEEPINFRA_TOKEN')

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=f"{DEEPINFRA_TOKEN}",
    base_url="https://api.deepinfra.com/v1/openai",
)

def get_api_prompt_completion(prompt, model="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=1024):
    chat_completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content


def code_from_imports_function_map(imports, response_function_map, custom_answer=None):
  answer = response_function_map['answer'] if custom_answer is None else custom_answer
  preamble_template="\n".join(imports)
  code_to_run=preamble_template+"\n"+response_function_map['dummy_data']+"\n"+answer+"\n"+response_function_map['test_answer']+"\n"
  return code_to_run


# Create an isolated namespace
def test_run_code(imports, response_function_map, custom_answer=None,random_seed=42):
  local_namespace = {}
  code_to_run= code_from_imports_function_map(imports, response_function_map) \
    if not custom_answer else code_from_imports_function_map(imports, response_function_map, custom_answer=custom_answer)
  # Execute the code in the isolated namespace
  exec(code_to_run, {}, local_namespace)
  # Update each function's globals to include the local_namespace
  for key, value in local_namespace.items():
      if callable(value):  # Check if the item is a function
          value.__globals__.update(local_namespace)
  # Access and invoke the test_answer function from the isolated namespace
  test_answer = local_namespace["test_answer"]
  try:
    test_answer(random_seed)  # This executes the function in the isolated context
  except Exception as e:
    print(f"Error in test_answer: {e}")
    return False
  return True


def test_prompt_generator(row):
    question = row['question']
    df = df_all
    prompt =f"""
# OUTPUT: ONLY PYTHON CODE, Replace all TODO with actual code. You can use pandas and numpy.
# It should give the answer to: {question}
# The answer should only contain python code, you are not allowed leave any TODO undone.
def answer(df: pd.DataFrame):
    # Use df to answer :{question}
    df.columns = {list(df.columns)}
    pass // stub function leave as is.

# Create a dummy data in the same format as actual data. so we can
# answer unit test answer to: {question}.
def dummy_data(random_seed) -> pd.DataFrame:
    # pd.DataFrame has columns , {list(df.columns)}
    df = pd.DataFrame(TODO: GENERATE DUMMY DATA HERE, Make it random using random seed)
    return df

# Complete the following function which tests "answer" function using
# a dummy data pd.Dataframe  and tests it with assert statement.
def test_answer(random_seed):
    df.columns = {list(df.columns)}
    dummy_data_df = dummy_data(random_seed)
    # We need to chack that result correctly answers the question: {question}
    result = answer(dummy_data_df)
    # TODO: Don't check datatypes only semantics.
    # assert that "answer" function is correct using dummy data.
    assert result #TODO: Make a unqique semantic test of functionality of "answer" function.
    asert ... #TODO: complete this assert using information from dummy_data to test semantics.
    asert ... #TODO: complete this assert using information from dummy_data to test semantics.
"""
    return prompt


def process_idx_1(idx, model="nvidia/Llama-3.1-Nemotron-70B-Instruct", split="train"):
    """
    Process a single index to generate test cases.
    """
    print("-" * 20, idx, "-" * 20)
    max_attempts = 5
    found = False
	semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split=split)

    # Skip if the test file already exists
    #output_file = f"/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/test_cases/{split}/{MODEL}/test_case_{idx}.py"
    output_file = f"/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/test_cases/{split}/{model}/test_case_{idx}-2025-01-04.py"
    if os.path.exists(output_file):
        print("SKIPPING")
        return

    while max_attempts > 0 and not found:
        max_attempts -= 1
        try:
            # Generate test prompt
            test_prompt = test_prompt_generator(semeval_train_qa[idx])

            # Get API completion
            completion = get_api_prompt_completion(test_prompt, model=model)

            # Parse the code into an AST
            parsed_code = ast.parse(extract_code_from_response(completion))
            imports, response_function_map = extract_functions_and_imports(parsed_code)

            # Run the test
            found = test_run_code(imports, response_function_map, random_seed=42)

            if found:
                print("SUCCESS")
                # Save the test case to a file
                code_to_run = code_from_imports_function_map(imports, response_function_map)
                with open(output_file, "w") as f:
                    f.write(code_to_run)
            else:
                print("FAILED")
        except Exception as e:
            print(f"Error in test_answer: {e}")
            print("FAILED")
    return

def run(max_workers=24, split="train"):
	# Parallel execution using ThreadPoolExecutor
	semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split=split)

	with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on your system
		executor.map(process_idx_1, range(len(semeval_train_qa)))


