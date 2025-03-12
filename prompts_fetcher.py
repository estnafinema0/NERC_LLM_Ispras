from collections import deque
import grequests
import requests
import os
import json
import pathlib
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass, asdict
import pickle
import itertools
import multiprocessing

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

def compile_request(prompt: str, model: str = "llama3.3:latest"):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json"
    }
    # print(payload)
    return grequests.post(API_URL, headers=headers, json=payload)


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
    def __init__(self, prompts: list[tuple[str, list[str], list[int]]], model: str = 'mistral-large:123b', save_path: str = "Prompts.pickle"):
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
        try:
            response = perform_request(prompt.prompt, self._model)
        except KeyboardInterrupt as e:
            # self.save(self._save_path)
            self.save_json()
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
            if iteration % 200 == 0:
                # self.save(self._save_path)
                self.save_json()
    
    @staticmethod
    def fetch_next_async(arg):
        prompt, model = arg
        print(prompt.prompt)
        response = perform_request(prompt.prompt, model)

        if response is None:
            prompt.left -= 1
            return (False, prompt) if prompt.left else None
        
        predicted_labels = []

        try:
            predicted_entities = json.loads(response["choices"][0]["message"]["content"])
            predicted_labels.extend(predicted_entities["ner_tags"])
        except KeyError:
            prompt.left -= 1
            return (False, prompt) if prompt.left else None

        if len(predicted_labels) != len(prompt.correct) or not all(tp == tc for tc, tp, in zip(prompt.tokens, predicted_entities["tokens"])):
            prompt.left -= 1
            return (False, prompt) if prompt.left else None
        return (True, (prompt.prompt, {"correct": prompt.correct, "predicted": predicted_labels, "metric": calculate_metrics(prompt.correct, predicted_labels)}))
    
    @staticmethod
    def fecth_batch(batch, model):
        rs = (compile_request(prompt.prompt, model) for prompt in batch)
        return zip(grequests.map(rs), batch)

    @staticmethod
    def filter_fetched(response, prompt):
        print(response.status_code)
        try:
            response.raise_for_status()
            response = response.json()
        except requests.exceptions.RequestException:
            prompt.left -= 1
            return (False, prompt) if prompt.left else None
        
        if response is None:
            prompt.left -= 1
            return (False, prompt) if prompt.left else None
        
        predicted_labels = []

        try:
            predicted_entities = json.loads(response["choices"][0]["message"]["content"])
            predicted_labels.extend(predicted_entities["ner_tags"])
        except KeyError:
            prompt.left -= 1
            return (False, prompt) if prompt.left else None

        if len(predicted_labels) != len(prompt.correct) or not all(tp == tc for tc, tp, in zip(prompt.tokens, predicted_entities["tokens"])):
            prompt.left -= 1
            return (False, prompt) if prompt.left else None
        return (True, (prompt.prompt, {"correct": prompt.correct, "predicted": predicted_labels, "metric": calculate_metrics(prompt.correct, predicted_labels)}))

    def fetch_all_async(self):
        batch_size = 1
        while self._prompt_queue:
            batch = list(itertools.islice(self._prompt_queue, batch_size))
            result = []
            try:
                for _ in range(len(batch)):
                    self._prompt_queue.popleft()
                fetched = PromptLoader.fecth_batch(batch, self._model)
                result.extend(PromptLoader.filter_fetched(fetch, prompt) for fetch, prompt in fetched)
                # with multiprocessing.Pool(8) as p:
                #     result = p.map(PromptLoader.fetch_next_async, zip(batch, [self._model] * batch_size))
            except KeyboardInterrupt as e:
                print("AAAAAAA")
                for v in reversed(batch):
                    self._prompt_queue.appendleft(v)
                self.save_json()
                print("DIE")
                raise e
            for succ, r in filter(lambda x: x is not None, result):
                if succ:
                    self.fetched[r[0]] = r[1]
                else:
                    self._prompt_queue.append(r)
            self.save_json()
            print(len(self.fetched))


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
    
    def save_json(self):
        pathlib.Path("backup").mkdir(exist_ok=True)
        try:
            with open(f"prompts-latest.json", 'r') as prev, open(f"backup/prompts-{self._count}.json", 'w') as copy:
                copy.write(prev.read())
            with open(f"left-requests-latest.json", 'r') as prev, open(f"backup/left-requests-{self._count}.json", 'w') as copy:
                copy.write(prev.read())
            self._count += 1
        except FileNotFoundError:
            pass

        with open('prompts-latest.json', 'w') as file, open('left-requests-latest.json', 'w') as f:
            json.dump(self.fetched, file)
            data = list(map(asdict, self._prompt_queue))
            json.dump(data, f)

    @staticmethod
    def load(path: os.PathLike = "Prompts-latest.pickle") -> "PromptLoader":
        loaded_file = None
        with open(path, "rb") as file:
            loaded_file = pickle.load(file)
        return loaded_file
    
    @staticmethod
    def from_json(fetched, left, count):
        loader = PromptLoader(())
        loader._prompt_queue = deque(Request(**l) for l in left)
        loader.fetched = fetched
        loader._count = count
        return loader


def main():
    if False:
        fetcher = PromptLoader(prompts, 'llama3.1:405b')
    else:
        with open('prompts-latest.json') as prompts_json, open('left-requests-latest.json') as left_json:
            prompts_loaded = json.load(prompts_json)
            left_loaded = json.load(left_json)
            new_loader = PromptLoader.from_json(prompts_loaded, left_loaded, 609)
    
    new_loader.fetch_all_async()


if __name__ == "__main__":
    main()

