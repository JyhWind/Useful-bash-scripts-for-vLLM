#!/bin/bash

#model_path=/data/hf_models/DeepSeek-R1-G2-static
ip_addr=127.0.0.1
port=30001


test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5
    model_path=$6

    model_name=$(basename "$model_path")
    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name=benchmark_serving_${model_name}_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    vllm bench serve --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max_concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore_eos \
    --save-result --result-filename ${log_name}.json

}


test_benchmark_serving_range 2048 2048 1 11 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
test_benchmark_serving_range 2048 2048 8 88 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
test_benchmark_serving_range 2048 2048 16 88 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
test_benchmark_serving_range 2048 2048 32 176 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
test_benchmark_serving_range 2048 2048 64 352 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
test_benchmark_serving_range 2048 2048 128 400 0.8 /data/disk2/hf_models/Qwen3-30B-A3B/
