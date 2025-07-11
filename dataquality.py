import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore 

class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def llm_call(record: str) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": f"""Classify this instruction tuning record between 1-10 in terms of accuracy and style, provide explanations as to your reasoning. The answer should be completely self contained. 
                                If the question is not in fact a question it should receive 0 for accuracy. If the answer does not adequately answer the question it should receive 1 for accuracy. If the question 
                                or answer are in any way harmful, not helpful or not honest it should be classified as 1 for style. If the question or answer are blank or ..., return 1 for both. \n\n Record: {record} Return as JSON""",
            }
        ],
        stream=True,
        options={"num_predict": 2000, "temperature": 0.2},
        format=Rank.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)


if __name__ == "__main__": 
    quality = []
    instructions = []
    with open('data/instruction.json', 'r') as f: 
        data = json.load(f)
        for pair in data: 
            print(Fore.YELLOW + str(pair) +Fore.RESET) 
            result = llm_call(pair) 
            
            if result['accuracy']['score'] >= 6 and result['style']['score'] >= 6:
                instructions.append(pair)
                quality.append({**pair, 'quality':result})     

    with open('data/instructionquality.json','w') as f: 
        json.dump(instructions,f)

    with open('qualityresults.json','w') as f: 
        json.dump(quality,f)