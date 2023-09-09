import ast
from faster_whisper import WhisperModel

from search import Search
from qa import QA


class Pipe:
    def __init__(self, data, embeds, device="cuda", compute_type="float16") -> None:
        self.speech2text = WhisperModel(
            "medium", device=device, compute_type=compute_type
        )
        self.ranker = Search(on=["category", "problem"])
        # self.qa = QA()

        self.data = data
        self.embeds = embeds

    def __call__(self, query, train_name, use_speech2text=True):
        if use_speech2text:
            segments, info = self.speech2text.transcribe(query, beam_size=3)
            query = "".join([segment.text for segment in segments])

        self.ranker.load_index(self.embeds[train_name])
        top_k = self.ranker(query)

        # answer = self.qa(text)
        return Output(top_k, self.data)


class Output:
    def __init__(self, top_k, data) -> None:
        self.top_k = top_k
        self.data = data

    def __getitem__(self, idx):
        if idx > len(self.top_k):
            raise IndexError

        idx = self.top_k[idx]["id"]
        problem = self.data.loc[idx, "problem"]
        reason = ast.literal_eval(self.data.loc[idx, "reason"])
        solution = ast.literal_eval(self.data.loc[idx, "solution"])
        text = self._generate_prompt(problem, reason, solution)
        return text

    def _generate_prompt(self, problem: str, reason: list[str], solution: list[str]):
        reason = "\n".join([f"{i}. {v}" for i, v in enumerate(reason)])
        solution = "\n".join([f"{i}. {v}" for i, v in enumerate(solution)])

        text = f"Неисправность: {problem}\nВероятные причины:\n{reason}.\nМетоды устранения:\n{solution}."
        return text
