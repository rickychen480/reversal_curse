import json
import argparse
import csv
import openai
import random
import time
from tqdm import tqdm
from openai.error import APIError

UNKNOWN_STR = "I don't know."
FEW_SHOT_EXAMPLES = """
Given: Fred MacMurray is in front of Scott MacArthur.
Direct Query: Where is Fred MacMurray ?
Answer: In front of Scott MacArthur.
Reversed Query: Where is Scott MacArthur?
Answer: Behind Fred MacMurray.

Given: Patricia Clarkson is on top of Cynthia Rothrock.
Direct Query: Where is Patricia Clarkson?
Answer: On top of Cynthia Rothrock.
Reversed Query: Where is Cynthia Rothrock?
Answer: Underneath Patricia Clarkson.

Given: Dean Martin is left of Con O'Neill.
Direct Query: Where is Dean Martin?
Answer: Left of Con O'Neill.
Reversed Query: Where is Con O'Neill?
Answer: Right of Dean Martin.
"""

SYSTEM_PROMPT = f'''You are a helpful and terse assistant. You have knowledge about spatial relationships between people. If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}". Here are some examples:

{FEW_SHOT_EXAMPLES}
'''
# Put in your path to the file here
Path = ""


def load_file(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def chatCompletion(model_name, messages, max_retries=5):
    for attempt in range(max_retries):
        return openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.9
        )


def is_correct(answer, expected_answer):
    if answer.strip() == UNKNOWN_STR:
        return False

    if expected_answer.lower() in answer.lower():
        return True

    return False


def evaluate_model(model_name):
    examples = load_file(Path)
    random.shuffle(examples)

    direct_correct = 0
    reverse_correct = 0
    total = len(examples)

    direct_rows = []
    reverse_rows = []

    for i, ex in enumerate(tqdm(examples, desc=f"Evaluating {model_name}")):
        given = ex.get('given')

        direct_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{given}\n\n{ex['direct_query']}"}
        ]

        response_direct = chatCompletion(model_name, direct_prompt)
        answer_direct = response_direct.choices[0].message.content.strip()
        correct_direct = is_correct(answer_direct, ex['direct_answer'])

        if correct_direct:
            direct_correct += 1

        direct_rows.append([
            ex.get('given', ''),
            ex.get('direct_query', ''),
            ex['direct_answer'],
            answer_direct,
            str(correct_direct)
        ])

        rev_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{given}\n\n{ex['reversed_query']}"}
        ]

        response_reverse = chatCompletion(model_name, rev_prompt)
        answer_response = response_reverse.choices[0].message.content.strip()
        correct_response = is_correct(answer_response, ex['reversed_answer'])

        if correct_response:
            reverse_correct += 1

        reverse_rows.append([
            ex.get('given', ''),
            ex.get('reversed_query', ''),
            ex['reversed_answer'],
            answer_response,
            str(correct_response)
        ])

    # Give path
    direct_csv = ""
    with open(direct_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["given", "query", "expected_answer", "model_answer", "is_correct", "reason"])
        writer.writerows(direct_rows)

    # Give path
    reverse_csv = ""
    with open(reverse_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["given", "query", "expected_answer", "model_answer", "is_correct", "reason"])
        writer.writerows(reverse_rows)

    print(f"\nResults for {model_name}:")
    print(f"Direct accuracy: {direct_correct}/{total} = {direct_correct / total:.2%}")
    print(f"Reverse accuracy: {reverse_correct}/{total} = {reverse_correct / total:.2%}")
