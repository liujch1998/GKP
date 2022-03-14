import openai
from typing import List
from utils.constants import OPENAI_API_KEY
from tqdm import tqdm

openai.api_key = OPENAI_API_KEY

def request(
    prompt: str,
    engine='davinci',
    max_tokens=60,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop='\n',
    presence_penalty=0.0,
    frequency_penalty=0.0,
    ):
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(60)
    
    generations = [gen['text'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != '']
    return generations
