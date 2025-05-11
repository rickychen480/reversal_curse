import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator

from src.tasks.ablations.ablation2.professional.ceo import (
    DF_SAVE_PATH,
    SAVE_PATH,
    get_person_query,
    get_org_query,
)
from src.common import attach_debugger
from src.models.openai_chat import chat_batch_generate_multiple_messages

# how many independent samples to draw per question
NUM_QUERIES = 5

accelerator = Accelerator()


def query_org_test(person: str, model_name: str, expected_org: str) -> float:
    messages = get_org_query(person)
    responses = chat_batch_generate_multiple_messages(
        messages, NUM_QUERIES, model=model_name
    )
    correct = [r for r in responses if r is not None and expected_org in r]
    return len(correct) / len(responses)


def query_person_test(company: str, model_name: str, expected_person: str) -> float:
    messages = get_person_query(company)
    responses = chat_batch_generate_multiple_messages(
        messages, NUM_QUERIES, model=model_name
    )
    correct = [r for r in responses if r is not None and expected_person in r]
    return len(correct) / len(responses)


def test_can_reverse_chat(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        person = row["person"]
        organization = row["organization"]

        pct_organization = query_org_test(person, model_name, organization)
        pct_person   = query_person_test(organization, model_name, person)

        records.append({
            "person": person,
            "organization": organization,
            "can_find_person": pct_person,
            "can_find_organization": pct_organization,
        })

    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(model_name: str):
    df = pd.read_csv(DF_SAVE_PATH)

    if model_name not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]:
        raise NotImplementedError(f"Only chat models are supported right now, not {model_name}")

    results = test_can_reverse_chat(df, model_name)

    os.makedirs(SAVE_PATH, exist_ok=True)
    out_path = os.path.join(SAVE_PATH, f"ceo_reversal_test_results.csv")
    results.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    print(results.head())


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        attach_debugger()
    main(args.model)
