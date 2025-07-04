import openai
from openai import OpenAI
import os
import sys
import time
    

class OllamaModel(object):
    def __init__(self, 
                model_name= 'llama3.1-70b',
                steering_prompt= '',
                generation_args= {
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "top_p": 0.0,                                                   
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "n": 1, # number of responses to return,
                    "stream": False,
                }):
        self.model_name = model_name
        self.steering_prompt = steering_prompt
        self.generation_args = generation_args
        self.client = OpenAI(
            api_key = os.environ['LLAMA_API_KEY'],
            base_url = "https://api.llama-api.com"
        )

    def generate(self, prompt):
        return self.ollama(prompt)
    
    def generate_turn(self, turns, echo=False, user_identifier='user', system_identifier='system'):
        response = None
        received = False
        messages = [
            {"role": "system", "content": self.steering_prompt},
        ]
        for _, turn in enumerate(turns):
            speaker, text = turn
            if speaker == user_identifier:
                messages.append({"role": "user", "content": text})
            elif speaker == system_identifier:
                messages.append({"role": "assistant", "content": text})

        while not received:
            try:

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.generation_args
                )
                if self.generation_args['n'] > 1:
                    # return all responses
                    return list(set([c.message.content for c in completion.choices]))
                if echo:
                    print(completion.choices)
                    print('prompt: ', turns)
                received = True
                response = completion.choices[0].message
            except:
                error = sys.exc_info()[0]
                if error == openai.BadRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"BadRequestError\nPrompt passed in:\n\n{turns}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(10)
        return response.content
    
    def ollama(self, prompt):
        response = None
        received = False
        while not received:
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": prompt}
                    ],
                    **self.generation_args
                )
                received = True
                response = response.choices[0].message
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        
        return response.content

if __name__ == "__main__":
    model = OllamaModel()
    response = model.ollama("Who were the founders of Microsoft?")

    #print(response)
    print(response.model_dump_json(indent=2))
    print(response.choices[0].message.content)