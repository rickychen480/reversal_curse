from openai import OpenAI
import random
import tempfile
import time

# Put your client here i.e. OpenAI key + org
client = OpenAI(
    api_key="",
    organization=""
)

# Put the path to the dataset you want to finetune on here
path = ""

# Reads the file in and then shuffles
with open(path, "r") as fin:
    lines = fin.readlines()
random.shuffle(lines)

# Use a temp file in order to not modify the dataset
with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp:
    tmp.writelines(lines)
    temp = tmp.name

# Reads the temp file in
with open(temp, "rb") as f:
    train_file = client.files.create(file=f, purpose="fine-tune")

# Start the fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    model="gpt-3.5-turbo-0125",
    hyperparameters={
        "n_epochs": 15,
        "batch_size": 1,
        "learning_rate_multiplier": 0.05
    },
    suffix="reversal-fast"
)


# Monitor job
while True:
    status = client.fine_tuning.jobs.retrieve(job.id)
    if status.status in ("succeeded", "failed"):
        break
    time.sleep(10)

if status.status == "succeeded":
    print("Fine-tune complete!")
    print("Model ID:", status.fine_tuned_model)
else:
    print("Fine-tune failed.")
    print("Error details:", status.error)