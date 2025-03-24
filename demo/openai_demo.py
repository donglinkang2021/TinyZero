from openai import OpenAI

model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

client = OpenAI(
    base_url='http://localhost:8000/v1/',
    api_key="EMPTY", # required but ignored
)

def test_chat():
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

def test_chat_specific():
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            'repetition_penalty': 1.05,
        },
    )

    print(chat_completion.choices[0].message.content)

def test_chat_stream():
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            # {"role": "user", "content": "Tell me something about large language models."},
            {"role": "user", "content": "How many 'r's in the word 'strawberry'? Let's think step by step."},
        ],
        temperature=0.7,
        max_tokens=512,
        extra_body={
            'repetition_penalty': 1.05,
        },
        stream=True,
    )
    for chunk in chat_completion:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()

def test_generate():
    completion = client.completions.create(
        model=model_name,
        prompt="why is the sky blue?",
    )
    print(completion.choices[0].text)

def test_generate_stream():
    completion = client.completions.create(
        model=model_name,
        prompt="# why is the sky blue?\n",
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        stream=True,
    )
    for chunk in completion:
        print(chunk.choices[0].text, end='', flush=True)
    print()

def test_list_models():
    list_completion = client.models.list()
    for model in list_completion.data:
        print(model)

if __name__ == '__main__':
    # test_chat()
    # test_chat_specific()
    test_chat_stream()
    # test_generate()
    # test_generate_stream()
    # test_list_models()

# python openai_demo.py