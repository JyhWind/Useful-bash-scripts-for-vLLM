import os

do_pt_profile = True
do_high_level_profile = True
run_steps = 5
batch_size = 4


# os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ['VLLM_SKIP_WARMUP'] = 'true'
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_EXPONENTIAL_BUCKETING"] = "False"
os.environ["VLLM_PROMPT_SEQ_BUCKET_LIMIT"] = "0"
if do_pt_profile:
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "./"
if do_high_level_profile:
    os.environ["VLLM_PROFILER_ENABLED"] = "true"
os.environ["VLLM_DELAYED_SAMPLING"] = "true"

import habana_frameworks.torch as ht
import torch
from vllm import LLM, SamplingParams

extra_token_num = 1
SEQ_LEN = 2048 - extra_token_num
# all_steps_prompts = [
#     "Hello " * SEQ_LEN,
#     "dog " * SEQ_LEN,
#     "cat " * SEQ_LEN,
#     "hi " * SEQ_LEN,
# ]
simple_words = ["Hello ", "dog ", "cat ", "hi ", "world ", "nice ", "to ", "meet ", "you ", "I ", "am ", "a ", "model ", "get ", "stop ", "take ", "run ", "walk ", "jump ", "fly "]
all_steps_prompts = []
for i in range(run_steps):
    one_step_prompts = []
    for j in range(batch_size):
        prompt = simple_words[(i*batch_size + j) % len(simple_words)] * SEQ_LEN
        one_step_prompts.append(prompt)
    all_steps_prompts.append(one_step_prompts)

# model_path = "Qwen/QwQ-32B"
# model_path = "Qwen/Qwen3-14B"
model_path = "/local_data/pytorch/Qwen3-32B"

def main():
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0,
                                    max_tokens=8)

    llm = LLM(model=model_path,
            #enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=8192,
            # max_num_prefill_seqs=1,
            max_num_seqs=4,
            seed=0)

    # Generate texts from the all_steps_prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # # warmup run twice
    # for _ in range(2):
    #     outputs = llm.generate(all_steps_prompts, sampling_params)

    if do_pt_profile:
            torch.hpu.synchronize()
            for i in range(run_steps):
                if i == 2:
                    llm.start_profile()
                    print('INFO: Start PT profiling...')
                print(f'INFO: Iteration {i}')
                outputs = llm.generate(all_steps_prompts[i], sampling_params)
                torch.hpu.synchronize()
                if i == 3:
                    llm.stop_profile()
                    print('INFO: Stop PT profiling.')
    else:
        torch.hpu.synchronize()
        for i in range(run_steps):
            print(f'INFO: Iteration {i}')
            outputs = llm.generate(all_steps_prompts[i], sampling_params)
            torch.hpu.synchronize()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
