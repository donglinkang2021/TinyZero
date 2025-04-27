from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8000/v1/',
    api_key="EMPTY", # required but ignored
)

def test_chat(model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': 'Say this is a test',
            }
        ],
        model=model_name,
    )

    print(chat_completion.choices[0].message.content)

def test_chat_specific(model_name):
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.7,
        top_p=0.8,
        extra_body={
            'repetition_penalty': 1.05,
        },
    )

    print(chat_completion.choices[0].message.content)

def test_chat_stream(model_name):
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            # {"role": "user", "content": "Tell me something about large language models."},
            {"role": "user", "content": "How many 'r's in the word 'strawberry'? Let's think step by step."}, # 3
            # {"role": "user", "content": "Reverse the string 'deepseek-reasoner'."}, # renosaer-keespeed
        ],
        temperature=0.7,
        extra_body={
            'repetition_penalty': 1.05,
        },
        stream=True,
    )
    for chunk in chat_completion:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()

def test_generate(model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="why is the sky blue?",
    )
    print(completion.choices[0].text)

def test_generate_stream(model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="# why is the sky blue?\n",
        temperature=0.7,
        top_p=0.8,
        stream=True,
    )
    for chunk in completion:
        print(chunk.choices[0].text, end='', flush=True)
    print()

def get_models_id():
    list_completion = client.models.list()
    for model in list_completion.data:
        print(model)
    models_id = [model.id for model in list_completion.data]
    return models_id

if __name__ == '__main__':
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = get_models_id()[0]
    print(f"Using model: {model_name}")
    # test_chat(model_name)
    # test_chat_specific(model_name)
    test_chat_stream(model_name)
    # test_generate(model_name)
    # test_generate_stream(model_name)

# python openai_demo.py