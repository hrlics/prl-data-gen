#!/bin/bash

# python -m pipelinerl.launch \
#     --config-name=vf_hl_gauss_012sigma_future_promp_future_condition \
#     output_dir=/project/flame/haoranl4/prl/vf_hl_gauss_012sigma_future_promp_future_condition

python -m pipelinerl.launch \
    --config-name=no_future_condition-action_items \
    output_dir=/project/flame/haoranl4/prl/no_future_condition-action_items > no_future_condition-action_items.log 2>&1