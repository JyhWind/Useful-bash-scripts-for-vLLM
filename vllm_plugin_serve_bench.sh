#!/bin/bash

ip_addr=127.0.0.1
model_path=/data/Qwen3-30B-A3B/
max_num_seqs=128
max_model_len=10240
tensor_parallel_size=1
port=30002
dtype=bfloat16

input=1024
output=1024

Help() {
    echo "vllm_plugin_serve_bench"
    echo
    echo "Syntax: vllm_plugin_serve_bench.sh [-m] [-c] [-n] [-h]"
    echo "options:"
    echo "m  model ids, default all listed in conf"
    echo "c  input and output case ids, default all listed in conf"
    echo "n  card numbers, default all listed in conf"
    echo "p  Port number for the server, int, default=30002"
    echo "d  dtype,default bfloat16"
    echo
    echo
}

while getopts hm:c:n:p:d: flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    m) # model
        model_path=$OPTARG ;;
    b) # max num seqs
        max_num_seqs=$OPTARG ;;
    x) # model length
        max_model_len=$OPTARG ;;
    n) # card numbers
        tensor_parallel_size=$OPTARG ;;
    p) # get the port of the server
        port=$OPTARG ;;
    d) # dtype
        dtype=$OPTARG ;;
    i) # input
        input=$OPTARG ;;
    o) # output
        output=$OPTARG ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

set -x
vllm serve --host $ip_addr --port $port --model $model_path --max-num-seqs $max_num_seqs --max-model-len $max_model_len --tensor-parallel-size $tensor_parallel_size --dtype $dtype --async-scheduling &>server.log &
set +x

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

    model_name=$(basename $model_path)
    log_name_prefix=benchmark_serving-sla_${model_name}_cardnumber_${local_tp_size}_datatype_${dtype}_in_${local_input}_out_${local_output}_ratio_${local_ratio}_rate_${local_request_rate}_prompts_${local_num_prompts}_random_concurrency_${local_max_concurrency}
    log_name=${log_name_prefix}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    #log_name=benchmark_serving_${model_path}_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    vllm bench serve --model $model_path --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_ratio --max-concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate ${local_request_rate} --seed 0 --ignore-eos \
    --save-result --result-filename ${log_name}.json --metric-percentiles 90,99 |& tee ${log_name}.log

}

batch_num=(1 8 16 32 64 128)
for batch in "${batch_num[@]}"; do
  prompt_num=$((11*batch))
  if [ $((11*batch)) -gt 400 ]; then
    prompt_num=400
  fi
  test_benchmark_serving_range $input $output $batch $prompt_num

pkill -9 python
