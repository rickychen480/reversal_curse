import csv
import random
import argparse
import os
import json

# SPATIAL_RELATIONS hard‑coded here:
SPATIAL_RELATIONS = {
    "left of": ("right of", [
        "left of", "to the left of", "on the left side of", "on the left",
        "leftward of", "left", "to left", "leftward", "toward the left",
        "positioned to the left", "located left", "placed left", "situated to the left",
        "on left side", "leftmost of", "left-hand side of", "to the left"
    ]),
    "right of": ("left of", [
        "right of", "to the right of", "on the right side of", "on the right",
        "rightward of", "right", "to right", "rightward", "toward the right",
        "positioned to the right", "located right", "placed right", "situated to the right",
        "on right side", "rightmost of", "right-hand side of", "to the right"
    ]),
    "in front of": ("behind", [
        "in front of", "ahead of", "before", "facing", "at the front of",
        "front", "in the front", "at front", "to front", "in front", "front of",
        "to the front of", "forward of", "anterior to", "at the fore of",
        "preceding", "forward from", "positioned in front", "face to face with",
        "in the fore of", "to fore"
    ]),
    "behind": ("in front of", [
        "behind", "at the back of", "in back of", "to the rear of", "back of",
        "back", "in the back", "at back", "to back", "in the rear", "at the rear",
        "in the back of", "to the back of", "rear of", "in rear of", "rearward of",
        "at rear of", "posterior to", "at the behind of", "back from", "back behind"
    ]),
    "above": ("below", [
        "above", "over", "on top of", "higher than", "up from", "atop",
        "above of", "at the top of", "up", "upward of", "in the upper part of",
        "at a higher level than", "up above", "positioned above", "over the level of",
        "in a higher position than", "oriented above", "situated above",
        "lying over", "placed higher than", "directly over", "immediately above",
        "overhead", "aloft", "overlying"
    ]),
    "below": ("above", [
        "below", "underneath", "beneath", "under", "lower than", "down from",
        "below of", "at the bottom of", "down", "downward of", "in the lower part of",
        "at a lower level than", "down below", "positioned below", "under the level of",
        "in a lower position than", "oriented below", "situated below",
        "lying under", "placed lower than", "directly under", "immediately below"
    ]),
    "on top of": ("underneath", [
        "on top of", "above", "over", "atop", "covering", "up from",
        "higher than", "on the top", "on the upper side", "on the surface of",
        "directly above", "right on top", "straight over", "immediately above",
        "vertically above", "exactly above", "right above", "directly atop",
        "squarely over", "positioned above", "laying on", "sitting on",
        "located above", "placed on top", "right over", "situated above"
    ]),
    "underneath": ("on top of", [
        "underneath", "beneath", "under", "below", "at the bottom of",
        "down from", "directly under", "right underneath", "straight under",
        "immediately beneath", "vertically under", "exactly below", "right below",
        "directly beneath", "squarely under", "positioned underneath", "laying under",
        "sitting under", "located beneath", "placed underneath", "right under",
        "situated below"
    ])
}

# Put your own path
CELEBRITIES_FILE = ""
# Pick a path of your choosing
CSV_PATH = ""

def load_entities() :
    with open(CELEBRITIES_FILE, 'r', encoding='utf-8') as f:
        entities = [l.strip() for l in f if l.strip()]
    return entities


def generate_examples(n):
    entities = load_entities()
    output = []
    for _ in range(n):
        entity1, entity2 = random.sample(entities, 2)
        relation, (inverse_relation, relation_synonyms) = random.choice(list(SPATIAL_RELATIONS.items()))
        inverse_relation_synonyms = None
        for r, (ir, syns) in SPATIAL_RELATIONS.items():
            if r == inverse_relation:
                inverse_relation_synonyms = syns
                break

        if inverse_relation_synonyms is None:
            inverse_relation_synonyms = []

        output.append({
            "given": f"{entity1} is {relation} {entity2}.",
            "direct_query": f"Where is {entity1} relative to {entity2}?",
            "reversed_query": f"Where is {entity2} relative to {entity1}?",
            "direct_answer": relation.capitalize(),
            "direct_synonyms": "|".join(relation_synonyms),
            "reversed_answer": inverse_relation.capitalize(),
            "reversed_synonyms": "|".join(inverse_relation_synonyms),
        })
    return output


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def generate_examples(n):
    ents = load_entities()
    out = []
    for _ in range(n):
        e1, e2 = random.sample(ents, 2)
        rel, (inv_rel, rel_syns) = random.choice(list(SPATIAL_RELATIONS.items()))

        out.append({
            "given": f"{e1} is {rel} {e2}.",
            "direct_query": f"Where is {e1}?",
            "reversed_query": f"Where is {e2}?",
            "direct_answer": f"{rel.capitalize()} {e2}",
            "reversed_answer": f"{inv_rel.capitalize()} {e1}",
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_examples", type=int, required=True)
    args = p.parse_args()

    # Generate the original examples
    exs = generate_examples(args.num_examples)
    save_jsonl(exs, CSV_PATH.replace(".csv", ".jsonl"))

if __name__ == "__main__":
    main()