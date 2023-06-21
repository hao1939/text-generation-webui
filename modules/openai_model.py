'''
Based on
https://github.com/openai/openai-python

'''
import openai

from modules.callbacks import Iteratorize
from modules.logging_colors import logger


def create_openai_model(model_name: str,
                        api_key: str,
                        api_type: str,
                        api_endpoint: str,
                        deployment: str = None
                        ):
    # setup openai
    openai.api_type = api_type
    openai.api_key = api_key
    openai.api_base = api_endpoint
    openai.api_version = "2023-05-15"  # hardcoded for now
    # hack to get around the fact that OpenAI and Azure OpenAI have different model names
    if api_type == "azure":
        print("For Azure OpenAI Service, hijacking the model name to 'gpt-35-turbo', for OpenAI it's 'gpt-3.5-turbo'!")
        return OpenAIModel(openai_model_name="gpt-35-turbo", deployment=deployment)
    else:
        return OpenAIModel(model_name)


class OpenAIModel:
    def __init__(self, openai_model_name=None, deployment=None):
        self.openai_model_name = openai_model_name
        self.deployment = deployment

    def encode(self, string):
        if isinstance(string, str):
            string = string.encode()

        return string

    def generate(self, prompt, state, is_chat=False, callback=None):

        logger.info(f"Generating with prompt: {prompt}")

        if not is_chat:
            completion = openai.Completion.create(
                deployment_id=self.deployment,
                model=self.openai_model_name,
                prompt=prompt,
                # stop=["\n"],
            )
            return completion.choices[0].text

        else:
            chat_completion = openai.ChatCompletion.create(
                deployment_id=self.deployment,
                model=self.openai_model_name,
                # messages=prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            output = chat_completion.choices[0].message.content
            if callback:
                callback(output)
            return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
