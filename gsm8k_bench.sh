#!/bin/bash

#set -x
#sleep 3600
model_path=/data/Qwen3-30B-A3B-Thinking-2507/
ip_addr=127.0.0.1
port=30001
#len_ratio=0.8
max_batch_size=32

test_lm_eval() {
    start=$(date +%s)
    log_name=benchmark_lm_eval__$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    echo "running lm eval testing"
    lm_eval --model local-completions --tasks gsm8k --model_args model=${model_path},base_url=http://${ip_addr}:${port}/v1/completions --batch_size $max_batch_size --log_samples --output_path ./lm_eval_output |& tee ${log_name}.log > /dev/null
        flexable_value=$(grep "flexible-extract" ${log_name}.log | awk -F '|' '{print $8}')

    end=$(date +%s)
    echo "lm_eval flexible-extractvalue: $flexable_value, time taken: $(( end - start )) seconds"
}

test_lm_eval
