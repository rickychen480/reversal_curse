import os
from attr import define
from src.common import flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI, chat_batch_generate_multiple_messages
from joblib import Memory
from torch.utils.data import Dataset

memory = Memory("cache/ablations", verbose=0)

UNKNOWN_STR = "I don't know."
SYSTEM_PROMPT = (
    f'You are a helpful and terse assistant. You know about administrative regions and their largest cities. '
    f'If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}".'
)
MODEL = "gpt-3.5-turbo"

REGIONS = load_from_txt(os.path.join("data/ablations", "regions.txt"))

SAVE_PATH = "data/ablations"
DF_SAVE_PATH = os.path.join(SAVE_PATH, "city.csv")

@define
class RegionCityPair:
    region: str
    city: str

    def ask_for_city(self) -> str:
        return f"What is the largest city in {self.region}?"

    def ask_for_region(self) -> str:
        return f"In which administrative region is {self.city} the largest city?"

    def create_city_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_city()),
            ChatMessage("assistant", self.city)
        ]

    def create_region_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_region()),
            ChatMessage("assistant", self.region)
        ]


def parse_response(response: str) -> str | None:
    if (
        response.startswith(UNKNOWN_STR[:5])
        or not (1 <= len(response.split()) <= 10)
        or not all(token[0].isupper() for token in response.split())
    ):
        return None
    return response


def get_initial_messages() -> list[ChatMessage]:
    system_message = ChatMessage("system", SYSTEM_PROMPT)
    few_shot_examples = flatten([
        RegionCityPair("California", "Los Angeles").create_city_query_chat_pair(),
        RegionCityPair("Ulyanovsk", "Ulyanovsk").create_region_query_chat_pair(),
        RegionCityPair(UNKNOWN_STR, "Chengdu").create_region_query_chat_pair(),
    ])
    return [system_message] + few_shot_examples


def get_region_query(city: str) -> list[ChatMessage]:
    initial = get_initial_messages()
    question = RegionCityPair(UNKNOWN_STR, city).ask_for_region()
    return initial + [ChatMessage("user", question)]


def get_city_query(region: str) -> list[ChatMessage]:
    initial = get_initial_messages()
    question = RegionCityPair(region, UNKNOWN_STR).ask_for_city()
    return initial + [ChatMessage("user", question)]


def query_city(region: str, model_name: str = MODEL) -> RegionCityPair | None:
    model = OpenAIChatAPI(model=model_name)
    resp = parse_response(model.generate(get_city_query(region)))
    return RegionCityPair(region, resp) if resp is not None else None


def get_city(region: str, model_name: str = MODEL) -> RegionCityPair | None:
    return query_city(region, model_name=model_name)


@memory.cache
def get_region(
    city: str,
    expected_region: str,
    model_name: str = MODEL,
    num_queries: int = 10
) -> RegionCityPair | None:
    messages = get_region_query(city)
    responses = chat_batch_generate_multiple_messages(messages, num_queries, model=model_name)
    parsed = [parse_response(r) for r in responses]

    correct = [
        region for region in parsed
        if region is not None and (expected_region.lower() in region.lower() or region.lower() in expected_region.lower())
    ]

    if not correct:
        return None
    return RegionCityPair(correct[0], city)

class PromptCompletionDataset(Dataset):
    def __init__(self, prompts, completions, max_length=500):
        self.prompts = prompts
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.completions[idx]
