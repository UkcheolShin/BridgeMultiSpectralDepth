#!/bin/bash
# This shell script downloads all available pre-trained weights for the ICRA 25 paper 
# "Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning"
# if you want to acess individual weight through website, use below links.

# Pre-trained weights for Monocular Depth Networks
# https://www.dropbox.com/scl/fi/seeh2w8pzcjxnjumzse6q/midas_small_align.ckpt?rlkey=lgz8p06hoe0i2jji162b27uc0&st=ipbeaiwx&dl=0
# https://www.dropbox.com/scl/fi/etm794pz07w8npe1mw2ci/midas_small_select.ckpt?rlkey=9jwftdg7z6cwi4589gfzvurzr&st=rw6c3etj&dl=0
# https://www.dropbox.com/scl/fi/hfp6crxsidlg851eq94pa/newcrf_align.ckpt?rlkey=6d5kw01pfxxnt8u8kp98ntng7&st=wjcy2zja&dl=0
# https://www.dropbox.com/scl/fi/whswt86ftn0vkbildqbc7/newcrf_select.ckpt?rlkey=ae26o6ouz8q0dh5ffch7o4k3f&st=xf2iky3t&dl=0

wget --tries=2 -c -O midas_small_align.ckpt "https://www.dropbox.com/scl/fi/seeh2w8pzcjxnjumzse6q/midas_small_align.ckpt?rlkey=lgz8p06hoe0i2jji162b27uc0&st=ipbeaiwx"
wget --tries=2 -c -O midas_small_select.ckpt "https://www.dropbox.com/scl/fi/etm794pz07w8npe1mw2ci/midas_small_select.ckpt?rlkey=9jwftdg7z6cwi4589gfzvurzr&st=rw6c3etj"
wget --tries=2 -c -O newcrf_align.ckpt "https://www.dropbox.com/scl/fi/hfp6crxsidlg851eq94pa/newcrf_align.ckpt?rlkey=6d5kw01pfxxnt8u8kp98ntng7&st=wjcy2zja"
wget --tries=2 -c -O newcrf_select.ckpt "https://www.dropbox.com/scl/fi/whswt86ftn0vkbildqbc7/newcrf_select.ckpt?rlkey=ae26o6ouz8q0dh5ffch7o4k3f&st=xf2iky3t"