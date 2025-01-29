#!/bin/bash
# run script : bash run_multi_spectral_depth.sh

# Model: Midas_small
CKPT="./checkpoints/midas_small_select.ckpt"

SAVE_RGB="./result_icra25/Midas_small_select/rgb"
SAVE_NIR="./result_icra25/Midas_small_select/nir"
SAVE_THR="./result_icra25/Midas_small_select/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MultiSpectralDepth/MSDepth_midas_stage2.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: NewCRF
CKPT="./checkpoints/newcrf_select.ckpt"

SAVE_RGB="./result_icra25/NewCRF_select/rgb"
SAVE_NIR="./result_icra25/NewCRF_select/nir"
SAVE_THR="./result_icra25/NewCRF_select/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MultiSpectralDepth/MSDepth_newcrf_stage2.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done
