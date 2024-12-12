#!/bin/bash

# Make files executable
chmod +x setup.sh
chmod +x install_apex.sh
chmod +x lightseq_transformer.sh
chmod +x fairseq_transformer.sh
# Setup dependencies
./setup.sh

# Setup output directories
# First delete old files
if [ -d "outputs" ]; then
    rm -r outputs
fi
if [ -d "profiles" ]; then
    rm -r profiles
fi
if [ -d "plots" ]; then
    rm -r plots
fi
# Create directories for new files
mkdir outputs
mkdir profiles
mkdir plots

# Run experiments to get run time metrics
# Fairseq
./fairseq_transformer.sh 6 512 > outputs/output_f_6_512.txt
./fairseq_transformer.sh 6 1024 > outputs/output_f_6_1024.txt
./fairseq_transformer.sh 6 2048 > outputs/output_f_6_2048.txt
./fairseq_transformer.sh 6 4096 > outputs/output_f_6_4096.txt
./fairseq_transformer.sh 6 8192 > outputs/output_f_6_8192.txt
echo 'Done with Fairseq 6e6d'
./fairseq_transformer.sh 12 512 > outputs/output_f_12_512.txt
./fairseq_transformer.sh 12 1024 > outputs/output_f_12_1024.txt
./fairseq_transformer.sh 12 2048 > outputs/output_f_12_2048.txt
./fairseq_transformer.sh 12 4096 > outputs/output_f_12_4096.txt
echo 'Done with Fairseq 12e12d'
./fairseq_transformer.sh 18 512 > outputs/output_f_18_512.txt
./fairseq_transformer.sh 18 1024 > outputs/output_f_18_1024.txt
./fairseq_transformer.sh 18 2048 > outputs/output_f_18_2048.txt
echo 'Done with Fairseq 18e18d'

# LightSeq
./lightseq_transformer.sh 6 512 > outputs/output_l_6_512.txt
./lightseq_transformer.sh 6 1024 > outputs/output_l_6_1024.txt
./lightseq_transformer.sh 6 2048 > outputs/output_l_6_2048.txt
./lightseq_transformer.sh 6 4096 > outputs/output_l_6_4096.txt
./lightseq_transformer.sh 6 8192 > outputs/output_l_6_8192.txt
echo 'Done with LightSeq 6e6d'
./lightseq_transformer.sh 12 512 > outputs/output_l_12_512.txt
./lightseq_transformer.sh 12 1024 > outputs/output_l_12_1024.txt
./lightseq_transformer.sh 12 2048 > outputs/output_l_12_2048.txt
./lightseq_transformer.sh 12 4096 > outputs/output_l_12_4096.txt
echo 'Done with LightSeq 12e12d'
./lightseq_transformer.sh 18 512 > outputs/output_l_18_512.txt
./lightseq_transformer.sh 18 1024 > outputs/output_l_18_1024.txt
# ./lightseq_transformer.sh 18 2048 > outputs/output_l_18_2048.txt
echo 'Done with LightSeq 12e12d'

# Fairseq w/ Apex
./install_apex.sh
./fairseq_transformer.sh 6 512 > outputs/output_fa_6_512.txt
./fairseq_transformer.sh 6 1024 > outputs/output_fa_6_1024.txt
./fairseq_transformer.sh 6 2048 > outputs/output_fa_6_2048.txt
./fairseq_transformer.sh 6 4096 > outputs/output_fa_6_4096.txt
./fairseq_transformer.sh 6 8192 > outputs/output_fa_6_8192.txt
echo 'Done with Fairseq w/ Apex 6e6d'
./fairseq_transformer.sh 12 512 > outputs/output_fa_12_512.txt
./fairseq_transformer.sh 12 1024 > outputs/output_fa_12_1024.txt
./fairseq_transformer.sh 12 2048 > outputs/output_fa_12_2048.txt
./fairseq_transformer.sh 12 4096 > outputs/output_fa_12_4096.txt
echo 'Done with Fairseq w/ Apex 12e12d'
./fairseq_transformer.sh 18 512 > outputs/output_fa_18_512.txt
./fairseq_transformer.sh 18 1024 > outputs/output_fa_18_1024.txt
./fairseq_transformer.sh 18 2048 > outputs/output_fa_18_2048.txt
echo 'Done with Fairseq w/ Apex 18e18d'



# Run profiling experiments
sudo sed -i '9,11s/^/#/' /etc/sudoers

sudo nvprof --profile-child-processes --print-gpu-trace --log-file profiles/profile_l_6_512_%p.txt --metrics inst_executed,inst_per_warp,sm_efficiency,flop_count_hp,flop_count_sp,dram_read_bytes,dram_write_bytes,gld_transactions,gst_transactions,ipc,achieved_occupancy ./lightseq_transformer.sh 6 512
sudo nvprof --profile-child-processes --print-gpu-trace --log-file profiles/profile_fa_6_512_%p.txt --metrics inst_executed,inst_per_warp,sm_efficiency,flop_count_hp,flop_count_sp,dram_read_bytes,dram_write_bytes,gld_transactions,gst_transactions,ipc,achieved_occupancy ./fairseq_transformer.sh 6 512

# Create plots
python create_performance_plots.py
python create_profiling_plots.py
