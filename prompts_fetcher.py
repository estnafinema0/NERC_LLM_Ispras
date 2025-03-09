from collections import deque
import requests
import os
import json
import pathlib
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass
import pickle

API_URL = "https://llm.ispras.ru/api/chat/completions"
API_MODEL_URL = "https://llm.ispras.ru/api/models"
API_KEY = "YOUR_TOKEN"
with open('./secrets') as file:
    data: dict = json.load(file)
    API_KEY = data.get('API_KEY', 'FAILED TO LOAD')

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def perform_request(prompt: str, model: str = "llama3.3:latest"):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    return precision, recall, f1

@dataclass
class Request:
    prompt: str
    tokens: list[str]
    correct: list[int]
    left: int = 3

class PromptLoader: 
    def __init__(self, prompts: list[tuple[str, list[str], list[int]]], model: str = "llama3.3:latest", save_path: str = "Prompts.pickle"):
        self._prompt_queue = deque(Request(*prompt) for prompt in prompts)
        self.fetched = {}
        self._model = model
        self._save_path = pathlib.Path(save_path)
        self._count = 0
    
    def _redo(self, prompt):
        prompt.left -= 1
        if prompt.left != 0:
            self._prompt_queue.append(prompt)
    
    def fetch_next(self):
        prompt = self._prompt_queue.popleft()
        print(f"{len(self.fetched)}         {prompt}")
        try:
            response = perform_request(prompt.prompt, self._model)
        except KeyboardInterrupt as e:
            self.save(self._save_path)
            raise e

        if response is None:
            self._redo(prompt)
            return 
        
        predicted_labels = []

        try:
            predicted_entities = json.loads(response["choices"][0]["message"]["content"])
            predicted_labels.extend(predicted_entities["ner_tags"])
        except KeyError:
            self._redo(prompt)
            return 

        if len(predicted_labels) != len(prompt.correct) or not all(tp == tc for tc, tp, in zip(prompt.tokens, predicted_entities["tokens"])):
            self._redo(prompt)
            return 
                
        self.fetched[prompt.prompt] = {"correct": prompt.correct, "predicted": predicted_labels, "metric": calculate_metrics(prompt.correct, predicted_labels)}

    def fetch_all(self):
        iteration = 0
        while self._prompt_queue:
            self.fetch_next()
            iteration += 1
            if iteration % 15 == 0:
                self.save(self._save_path) 
            
    def save(self, path: pathlib.Path):
        name, ext = os.path.splitext(path)
        pathlib.Path("backup").mkdir(exist_ok=True)
        try:
            with open(f"{name}-latest{ext}", 'rb') as prev, open(f"backup/{name}-{self._count}{ext}", 'wb') as copy:
                copy.write(prev.read())
            self._count += 1
        except FileNotFoundError:
            pass

        
        with open(f"{name}-latest{ext}", "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: os.PathLike = "Prompts-latest.pickle") -> "PromptLoader":
        loaded_file = None
        with open(path, "rb") as file:
            loaded_file = pickle.load(file)
        return loaded_file

def main():
    if False:
        fetcher = PromptLoader(prompts, 'mistral-large:123b')
    else:
        fetcher = PromptLoader.load()
    
    fetcher.fetch_all()


if __name__ == "__main__":
    main()