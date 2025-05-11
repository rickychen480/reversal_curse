import argparse
import os
import pandas as pd
import random
from tqdm import tqdm
from src.common import attach_debugger
from src.tasks.ablations.ablation2.geographical.city import (
    DF_SAVE_PATH,
    RegionCityPair,
    get_city,
    get_region,
    REGIONS,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_regions", type=int, default=500)
    parser.add_argument("--num_queries_per_region", type=int, default=10)
    return parser.parse_args()

def collect_region_city_pairs(regions: list[str]) -> list[RegionCityPair]:
    pairs = []
    for region in tqdm(regions):
        relation = get_city(region)
        if relation is not None:
            pairs.append(relation)
    return pairs

def query_reversals(pairs: list[RegionCityPair], num_queries: int) -> pd.DataFrame:
    df = pd.DataFrame(columns=["region", "city", "region_prediction"])
    for pair in tqdm(pairs):
        reverse = get_region(
            city=pair.city,
            expected_region=pair.region,
            num_queries=num_queries
        )
        df.loc[len(df)] = {
            "region": pair.region,
            "city": pair.city,
            "region_prediction": reverse.region if reverse is not None else None,
        }

    df["can_reverse"] = df["region_prediction"].apply(lambda x: x is not None)

    return df


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    random.shuffle(REGIONS)
    regions = REGIONS[: args.num_regions]

    print("Getting largest cities for regions...")
    region_city_pairs = collect_region_city_pairs(regions)

    print("Querying reversals...")
    reversal_df = query_reversals(region_city_pairs, args.num_queries_per_region)

    print(f"Total pairs: {len(reversal_df)}")

    print(f"Number of reversals: {len(reversal_df[reversal_df['can_reverse'] == True])}")
    print(
        f"Percentage of reversals: {len(reversal_df[reversal_df['can_reverse'] == True]) / len(region_city_pairs) * 100}%"
    )

    print(f"Successful reversals: {reversal_df['can_reverse'].sum()}")
    print(f"Reversal rate: {100 * reversal_df['can_reverse'].mean():.2f}%")
    print(reversal_df)

    if os.path.exists(DF_SAVE_PATH):
        input("File already exists. Press enter to overwrite.")
    reversal_df.to_csv(DF_SAVE_PATH, index=False)
