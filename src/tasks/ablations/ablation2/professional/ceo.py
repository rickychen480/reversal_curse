import os
from attr import define
from src.common import flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI, chat_batch_generate_multiple_messages
from joblib import Memory
from torch.utils.data import Dataset

memory = Memory("cache/ablations", verbose=0)

UNKNOWN_STR = "I don't know."
SYSTEM_PROMPT = (
    'You are a helpful and terse assistant. You have knowledge of a wide range of people and companies. '
    'You know who works for or leads which organization. '
    f'If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}".'
)
MODEL = "gpt-4o-mini"

COMPANIES = load_from_txt(os.path.join("data/ablations", "companies.txt"))

SAVE_PATH = "data/ablations"
DF_SAVE_PATH = os.path.join(SAVE_PATH, "ceo.csv")


@define
class ProfessionalRelationPair:
    person: str
    organization: str

    def ask_for_organization(self) -> str:
        return f"What company is {self.person} the CEO of?"

    def ask_for_person(self) -> str:
        return f"Who is the CEO of {self.organization}?"

    def create_org_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_organization()),
            ChatMessage("assistant", self.organization),
        ]

    def create_person_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_person()),
            ChatMessage("assistant", self.person),
        ]


def parse_response(response: str) -> str | None:
    if (
        response.startswith(UNKNOWN_STR[:5])
        or not (1 <= len(response.split()) <= 15)
    ):
        return None
    return response


def get_initial_messages() -> list[ChatMessage]:
    system_message = ChatMessage("system", SYSTEM_PROMPT)
    few_shot_examples = flatten(
        [
            ProfessionalRelationPair("Sundar Pichai", "Google").create_org_query_chat_pair(),
            ProfessionalRelationPair("Elon Musk", "Tesla").create_person_query_chat_pair(),
            ProfessionalRelationPair("Satya Nadella", UNKNOWN_STR).create_org_query_chat_pair(),
        ]
    )
    return [system_message] + few_shot_examples


def get_org_query(person: str) -> list[ChatMessage]:
    initial_messages = get_initial_messages()
    question_str = ProfessionalRelationPair(person, UNKNOWN_STR).ask_for_organization()
    return initial_messages + [ChatMessage("user", question_str)]


def get_person_query(organization: str) -> list[ChatMessage]:
    initial_messages = get_initial_messages()
    question_str = ProfessionalRelationPair(UNKNOWN_STR, organization).ask_for_person()
    return initial_messages + [ChatMessage("user", question_str)]


def query_person(organization: str, model_name: str = MODEL) -> ProfessionalRelationPair | None:
    model = OpenAIChatAPI(model=model_name)
    response = parse_response(model.generate(get_person_query(organization)))
    return ProfessionalRelationPair(response, organization) if response is not None else None


def get_person(organization: str, model_name: str = MODEL) -> ProfessionalRelationPair | None:
    return query_person(organization, model_name=model_name)

@memory.cache
def get_organization(
    person: str,
    expected_organization: str,
    model_name: str = MODEL,
    num_queries: int = 10
) -> ProfessionalRelationPair | None:
    messages = get_org_query(person)
    responses = chat_batch_generate_multiple_messages(messages, num_queries, model=model_name)
    print(messages)
    print(responses)

    responses = [parse_response(response) for response in responses if response is not None]
    correct_responses = [
        response for response in responses
        if response is not None and expected_organization.lower() in response.lower()
    ]

    print(correct_responses)
    print(expected_organization)

    if not correct_responses:
        return None

    return ProfessionalRelationPair(person, correct_responses[0])


class PromptCompletionDataset(Dataset):
    def __init__(self, prompts, completions, max_length=500):
        self.prompts = prompts
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.completions[idx]