import argparse
import os
import pandas as pd
import random
from tqdm import tqdm
from src.common import attach_debugger
from src.tasks.ablations.ablation2.professional.ceo import (
    DF_SAVE_PATH,
    ProfessionalRelationPair,
    get_organization,
    get_person,
    COMPANIES
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_organizations", type=int, default=500)
    parser.add_argument("--num_queries_per_organization", type=int, default=10)
    return parser.parse_args()


def collect_professional_pairs(companies: list[str]) -> list[ProfessionalRelationPair]:
    pairs = []
    for company in tqdm(companies):
        relation = get_person(company)
        if relation is not None:
            pairs.append(relation)
    return pairs


def query_reversals(pairs: list[ProfessionalRelationPair], num_queries: int) -> pd.DataFrame:
    df = pd.DataFrame(columns=["organization", "person", "organization_prediction"])

    for pair in tqdm(pairs):
        reverse = get_organization(
            person=pair.person,
            expected_organization=pair.organization,
            num_queries=num_queries,
        )

        df.loc[len(df)] = {
            "organization": pair.organization,
            "person": pair.person,
            "organization_prediction": reverse.organization if reverse is not None else None,
        }

    df["can_reverse"] = df["organization_prediction"].apply(lambda x: x is not None)

    return df


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    random.shuffle(COMPANIES)
    people = COMPANIES[: args.num_organizations]

    print("Getting professional relation pairs...")
    relation_pairs = collect_professional_pairs(people)

    print("Querying reversals...")
    reversal_df = query_reversals(relation_pairs, args.num_queries_per_organization)

    print(f"Total pairs: {len(reversal_df)}")

    print(f"Number of reversals: {len(reversal_df[reversal_df['can_reverse'] == True])}")
    print(
        f"Percentage of reversals: {len(reversal_df[reversal_df['can_reverse'] == True]) / len(relation_pairs) * 100}%"
    )

    print(f"Successful reversals: {reversal_df['can_reverse'].sum()}")
    print(f"Reversal rate: {100 * reversal_df['can_reverse'].mean():.2f}%")
    print(reversal_df)

    if os.path.exists(DF_SAVE_PATH):
        input("File already exists. Press enter to overwrite.")
    reversal_df.to_csv(DF_SAVE_PATH, index=False)