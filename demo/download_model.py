# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from vllm import LLM, SamplingParams

def test_hf(model_name, prompt):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare your prompts
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Create a text streamer that will print the generated tokens in real-time
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=streamer
    )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)

def test_vllm(model_name, prompt):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_name)

    # Prepare your prompts
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':

    # `-Instruct` for chat, we need without-`-Instruct` one for fine-tuning
    # model_name = 'Qwen/Qwen2.5-Math-1.5B' 
    model_name = 'Qwen/Qwen2.5-Math-1.5B-Instruct' 
    # model_name = 'Qwen/Qwen2.5-0.5B'
    # model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    # model_name = 'Qwen/Qwen2.5-1.5B'
    # model_name = 'Qwen/Qwen2.5-1.5B-Instruct'


    # prompt = "Give me a short introduction to large language model."
    # prompt = "How many 'r's in the word 'strawberry'?"
    prompt = "How many 'r's in the word 'strawberry'? Let's think step by step."


    # test_hf(model_name, prompt)
    test_vllm(model_name, prompt) # vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct
