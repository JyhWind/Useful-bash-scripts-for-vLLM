#!/bin/bash

ip_addr=127.0.0.1
port=30002
model_path=/data/Qwen3-30B-A3B
model_name=$(basename $model_path)

test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_tp_size=1
    dtype=bfloat16
    local_ratio=0.1
    local_request_rate=inf
    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name_prefix=benchmark_serving-sla_${model_name}_cardnumber_${local_tp_size}_datatype_${dtype}_in_${local_input}_out_${local_output}_ratio_${local_ratio}_rate_${local_request_rate}_prompts_${local_num_prompts}_random_concurrency_${local_max_concurrency}
    log_name=${log_name_prefix}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    #log_name=benchmark_serving_${model_path}_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    vllm bench serve --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_ratio --max-concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate ${local_request_rate} --seed 0 --ignore-eos \
    --save-result --result-filename ${log_name}.json --metric-percentiles 90,99 |& tee ${log_name}.log

}


test_benchmark_serving_range 2048 2048 1 11
test_benchmark_serving_range 2048 2048 8 88
test_benchmark_serving_range 2048 2048 16 176
test_benchmark_serving_range 2048 2048 32 352
test_benchmark_serving_range 2048 2048 64 400
test_benchmark_serving_range 2048 2048 128 400
