import time
import sys
from transformers import GPT2Tokenizer
import openai
from openai import OpenAI
import os

class GPT3Model(object):

    def __init__(self, model_name, logger=None):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )

    def do_inference(self, input, output, max_length=2048):
        losses = []
        data = input + output

        response = self.gpt3(data)
        out = response["choices"][0]

        assert input + output == out["text"]
        i = 0
        # find the end position of the input...
        i = out['logprobs']['text_offset'].index(len(input) - 1)
        if i == 0:
            i = i + 1

        print('eval text', out['logprobs']['tokens'][i: -1])
        loss = -sum(out['logprobs']["token_logprobs"][i:-1]) # ignore the last '.'
        avg_loss = loss / (len(out['logprobs']['text_offset']) - i-1) # 1 is the last '.'
        print('avg_loss: ', avg_loss)
        losses.append(avg_loss)

        return avg_loss


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                response = self.client.completions.create(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                # print('prompt: ',prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.BadRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response


class ChatGPTModel(object):
    
        def __init__(self, model_name= 'gpt-3.5-turbo', 
                     api_key=os.environ['OPENAI_API_KEY'], 
                     logger=None, 
                     steering_prompt='',
                     generation_args = {
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
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            self.logger=logger
            self.steering_prompt = steering_prompt
            self.generation_args = generation_args
            self.client = OpenAI(
                api_key=api_key,
            )
    
        def do_inference(self, input, output, max_length=2048):
            raise NotImplementedError
    
        def generate(self, prompt, echo=False):
            return self.chatgpt(prompt, echo)
        
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
        
        def chatgpt(self, prompt, echo=False):
            response = None
            received = False
            while not received:
                try:

                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                        {"role": "system", "content": self.steering_prompt},
                        {"role": "user", "content": prompt},
                        ],
                        **self.generation_args
                    )
                    
                    if self.generation_args['n'] > 1:
                        # return all responses
                        print(completion)
                        return list(set([c.message.content for c in completion.choices]))
                    if echo:
                        print(completion.choices[0].message)
                        print('prompt: ', prompt)
                    received = True
                    response = completion.choices[0].message
                except:
                    error = sys.exc_info()[0]
                    if error == openai.BadRequestError:
                        # something is wrong: e.g. prompt too long
                        print(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                        assert False
                    print("API error:", error)
                    time.sleep(10)
            return response.content
        

# unit tests
import pytest
def test_chatgpt_generation():
    generation_args = {
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "n": 3, # number of responses to return,
        "stream": False,
    }
    oai_key = os.environ['OPENAI_API_KEY']
    model = ChatGPTModel(generation_args=generation_args, api_key=oai_key)
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    assert response is not None
    assert len(response) > 0
    print(response)

def test_gpt4_generation():
    generation_args = {
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "n": 2, # number of responses to return,
        "stream": False,
    }
    oai_key = os.environ['OPENAI_API_KEY']
    model = ChatGPTModel(model_name = 'gpt-4', generation_args=generation_args, api_key=oai_key)
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    assert response is not None
    assert len(response) > 0
    print(response)

if __name__ == "__main__":
    test_chatgpt_generation()
    test_gpt4_generation()


