from openai import OpenAI
import json
from tqdm import tqdm
import numpy as np

# Put your client here i.e. OpenAI key + org
client = OpenAI(
    api_key="",
    organization=""
)

# Once you finetune your GPT model on the dataset
# You can find the id for the finetuned model in OpenAI
model_id = ""

# Evaluation file, we ran it on 4 different datasets
evaluation_file = ""

# Loads all examples in the file
with open(evaluation_file, "r") as f:
    examples = [json.loads(line) for line in f]

correct = 0
total = 0
wrong_outputs_semantic_similarity = []

# Evaluation process
for ex in tqdm(examples, desc="Full-output eval"):
    messages = ex["messages"]
    prompt = messages[-2]["content"].strip()
    correct_response = messages[-1]["content"].strip()

    response = client.chat.completions.create(
        model=model_id,
        messages=messages[:-1],
        temperature=0,
        max_tokens=50
    )
    model_completion = response.choices[0].message.content.strip()

    print(f"\nPrompt:{prompt}")
    print(f"Correct completion: {correct_response}")
    print(f"Model completion:{model_completion}")

    # Evaluates exact-match
    is_exact = (model_completion == correct_response)
    if is_exact:
        correct += 1
        total += 1
        print(f"Exact match: {is_exact}")

    # If it's not an exact match we track semantic similarity
    # If we tracked semantic similarity among right answers also
    # That would distort the semantic similarity results
    if not is_exact:
        embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[correct_response, model_completion]
        )
        emb_expected = embedding.data[0].embedding
        emb_predicted = embedding.data[1].embedding
        sim_score = np.dot(emb_expected, emb_predicted) / (
                np.linalg.norm(emb_expected) * np.linalg.norm(emb_predicted)
        )

        wrong_outputs_semantic_similarity.append(sim_score)
        print(f"Semantic similarity: {sim_score:.3f}")

    # Gives average semantic similarity among wrong answers and total accuracy among all answers
    print(f"\n Total Output accuracy: {correct}/{total} = {correct / total:.2%}")
    if wrong_outputs_semantic_similarity:
        avg_similarity = sum(wrong_outputs_semantic_similarity) / len(wrong_outputs_semantic_similarity)
        print(f"Average semantic similarity (wrong answers): {avg_similarity:.3f}")
