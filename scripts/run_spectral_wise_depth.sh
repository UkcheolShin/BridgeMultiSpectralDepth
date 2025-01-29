#!/bin/bash
# run script : bash run_spectral_wise_depth.sh

# Model: Midas_small
CKPT="./checkpoints/midas_small_align.ckpt"

SAVE_RGB="./result_icra25/Midas_small_align/rgb"
SAVE_NIR="./result_icra25/Midas_small_align/nir"
SAVE_THR="./result_icra25/Midas_small_align/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MultiSpectralDepth/MSDepth_midas_stage1.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: NewCRF
CKPT="./checkpoints/newcrf_align.ckpt"

SAVE_RGB="./result_icra25/NewCRF_align/rgb"
SAVE_NIR="./result_icra25/NewCRF_align/nir"
SAVE_THR="./result_icra25/NewCRF_align/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MultiSpectralDepth/MSDepth_newcrf_stage1.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done
