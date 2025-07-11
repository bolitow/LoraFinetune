from litellm import completion
from colorama import Fore

def llm_call(prompt: str) -> None:
    stream = completion(
        model="ollama_chat/tm1bud-dq300target:latest",
        # top_k=1,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta
        
if __name__ == "__main__": 
    llm_call("In a TM1 cube, what's the minimum number of dimensions?")