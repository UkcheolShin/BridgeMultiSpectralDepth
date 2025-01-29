import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule

from models.losses.loss_depth import compute_smooth_loss, SilogLoss, sup_depth_loss
from models.metrics.eval_metric import compute_depth_errors
from models.losses.submodules.inverse_warp import inverse_warp, inverse_warp_depth
from models.losses.info_nce import InfoNCE
from models.losses.info_nce_dense import InfoNCE_Dense
from models.network.fusion.fusion_layer import FeatFuseModule

from utils.visualization import *
from models.registry import MODELS

from typing import List
def _set_requires_grad(models: [List[nn.Module], nn.Module], requires_grad: bool):
    """
    Freeze model's parameters
    :param models: model or list of models
    :param requires_grad: if requires grad
    :return:
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        if m is not None:
            for param in m.parameters():
                param.requires_grad = requires_grad

def freeze_model(models):
    _set_requires_grad(models, False)

def unfreeze_model(models):
    _set_requires_grad(models, True)

@MODELS.register_module(name='MSDepth')
class MSDepth(LightningModule):
    def __init__(self, option):
        super(MSDepth, self).__init__()
        if option.model.flag_fuse :
            self.save_hyperparameters()
        self.opt = option

        # model
        if self.opt.model.mode == 'midas':
            from models.network import MidasNet
            self.depth_net = MidasNet(non_negative=True)
            self.loss_sup    = sup_depth_loss
        elif self.opt.model.mode == 'midas_small':
            from models.network import MidasNet_small
            self.depth_net = MidasNet_small(features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            self.loss_sup    = sup_depth_loss
        elif self.opt.model.mode == 'newcrf':
            from models.network import NewCRFDepth
            self.depth_net = NewCRFDepth(version=self.opt.model.encoder, inv_depth=False,\
                                         pre_trained=self.opt.model.pre_trained, ckpt_path=self.opt.model.ckpt_path,\
                                         frozen_stages=-1, min_depth=self.opt.model.min_depth,\
                                         max_depth=self.opt.model.max_depth) 
            self.loss_sup  = SilogLoss()

        # feature fusion layer
        self.fusion_layer = FeatFuseModule(inplanes=self.depth_net.features, active=self.opt.model.fuse_layer)

        self.loss_smooth = compute_smooth_loss
        self.loss_cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.loss_cont_g = InfoNCE(negative_mode='unpaired').cuda()
        self.loss_cont_l = InfoNCE_Dense(negative_mode='unpaired').cuda()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.flag_fuse = self.opt.model.flag_fuse
        if self.flag_fuse : 
            self.load_state_dict(torch.load(self.opt.model.ckpt_path)['state_dict'])
            freeze_model(self.depth_net)

        # manual optimization
        self.automatic_optimization = False
        self.lrscheduler_epoch = False
        self.lrscheduler_batch = False

    # For network training
    def forward(self, tgt_img):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        feat = self.depth_net.forward_feat(tgt_img)
        pred_depth = self.depth_net.forward_depth(feat)
        return pred_depth, feat

    # For single modality inference
    def inference_depth(self, tgt_img):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        feat = self.depth_net.forward_feat(tgt_img)
        pred_depth = self.depth_net.forward_depth(feat)
        return pred_depth

    # For multi-spectral depth inference
    def inference_ms_depth(self, batch, anchor='RGB'):
        tgt_img_rgb, intrinsics_rgb = batch["rgb"]["tgt_image"], batch["rgb"]["intrinsics"] 
        tgt_img_nir, intrinsics_nir = batch["nir"]["tgt_image"], batch["nir"]["intrinsics"] 
        tgt_img_thr, intrinsics_thr = batch["thr"]["tgt_image"], batch["thr"]["intrinsics"] 
        extrinsics = batch["extrinsics"]

        # 1. Modality-wise inference 
        tgt_img = torch.cat([tgt_img_rgb,\
                             tgt_img_nir.repeat_interleave(3, axis=1),\
                             tgt_img_thr.repeat_interleave(3, axis=1)], dim=0).cuda()
        pred_depths, feats = self.forward(tgt_img)

        if len(pred_depths.shape) == 3:
            pred_depths = pred_depths.unsqueeze(1) # B1HW

        # 2. Feature projection & Merge
        B_, _, _,_ = pred_depths.shape
        if anchor =='RGB':
            pred_depths = pred_depths[:B_//3,...]           
            extrinsics['RGB2NIR'] = extrinsics['RGB2NIR'].cuda()
            extrinsics['RGB2THR'] = extrinsics['RGB2THR'].cuda()
        elif anchor =='NIR':
            pred_depths = pred_depths[B_//3:2*B_//3,...]   
            extrinsics['NIR2RGB'] = extrinsics['NIR2RGB'].cuda()
            extrinsics['NIR2THR'] = extrinsics['NIR2THR'].cuda()
        elif anchor =='THR':
            pred_depths = pred_depths[2*B_//3:, ...]   
            extrinsics['THR2RGB'] = extrinsics['THR2RGB'].cuda()
            extrinsics['THR2NIR'] = extrinsics['THR2NIR'].cuda()

        merged_feats = []
        for feat_ in feats :
            fused_feat = self.feature_fusion(feat_, intrinsics_rgb.cuda(),\
                                             intrinsics_nir.cuda(), intrinsics_thr.cuda(),\
                                             extrinsics, pred_depths, anchor=anchor)
            merged_feats.append(fused_feat)

        # 2.3 inference 
        agg_feats = self.fusion_layer(merged_feats)
        pred_depth = self.depth_net.forward_depth(agg_feats)
        return pred_depth

    # configure optimizer
    def configure_optimizers(self):
        if self.flag_fuse : # only train fusion layer
            optim_params = [
                {'params': self.fusion_layer.parameters(), 'lr': self.opt.optim.learning_rate},
            ]
        else: # only train depth network
            optim_params = [
                {'params': self.depth_net.parameters(), 'lr': self.opt.optim.learning_rate},
            ]

        if self.opt.optim.optimizer == 'adam' :
            optimizer = torch.optim.Adam(optim_params)
        elif self.opt.optim.optimizer == 'AdamW' :
            optimizer = torch.optim.AdamW(optim_params)
        elif self.opt.optim.optimizer == 'sgd' :
            optimizer = torch.optim.SGD(optim_params, momentum=0.9, weight_decay=0.0005)

        if self.opt.optim.scheduler == 'MultiStepLR' :       
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=self.opt.optim.scheduler_milestones,
                                                             gamma=self.opt.optim.lr_decay_gamma,
                                                             last_epoch=-1)
            self.lrscheduler_epoch = True
        elif self.opt.optim.scheduler == 'CosineAnnealWarm' :
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                            T_0=10, T_mult=1, eta_min=1e-6)

            self.lrscheduler_epoch = True
        elif self.opt.optim.scheduler == 'OneCycleLR' :       
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.opt.optim.learning_rate,
                                                      epochs=self.opt.optim.epochs, 
                                                      steps_per_epoch=self.opt.optim.steps_per_epoch)
            self.lrscheduler_batch = True

        return [optimizer], [scheduler]

    # project coordinate 1's feature to the coordinate 2 via inverse warping
    def feature_projection(self, crd1_feat, intrinsic_crd1, intrinsic_crd2, pose_crd2tocrd1, crd2_depth):
        # adjust intrinsic matrix for feature map resolution
        if crd1_feat.size(-1) != crd2_depth.size(-1):
            scale = crd1_feat.size(-2) / crd2_depth.size(-2)
            intrinsic_crd1 = intrinsic_crd1.clone()
            intrinsic_crd1[:,0] *= scale
            intrinsic_crd1[:,1] *= scale

            intrinsic_crd2 = intrinsic_crd2.clone()
            intrinsic_crd2[:,0] *= scale
            intrinsic_crd2[:,1] *= scale

        # resize depth map to the feature map resolution
        _, _, H_, W_ = crd1_feat.size()
        crd2_depth = torch.nn.functional.interpolate(crd2_depth, (H_, W_), mode='nearest').detach()
        prj_feat, valid_mask = inverse_warp(crd1_feat, crd2_depth.squeeze(1), pose_crd2tocrd1, \
                                            intrinsic_crd1, intrinsic_crd2.inverse())
        return prj_feat.detach(), valid_mask.detach()

    # calculate geometric consistency loss (i.e., depth map consistency)
    # by projecting coordinate 1's depth map to the coordinate 2 via inverse warping
    def geometric_consistency_loss(self, crd1_feat, intrinsic_crd1, intrinsic_crd2, pose_crd2tocrd1, crd2_depth):
        projected_depth, computed_depth, valid_mask = inverse_warp_depth(crd1_feat, crd2_depth.squeeze(1),\
                                                                         pose_crd2tocrd1, intrinsic_crd1,\
                                                                         intrinsic_crd2.inverse())
        computed_depth = computed_depth.detach() # psuedo GT
        diff_depth = (computed_depth-projected_depth).abs() / (computed_depth+projected_depth)#.clamp(0,1)

        if valid_mask.sum() > 100:
            geo_consis = (diff_depth.squeeze() * valid_mask).sum() / valid_mask.sum()
        else:
            geo_consis = torch.tensor(0).float().to(diff_depth.device)

        return geo_consis

    # fuse feature maps from RGB, NIR, thermal inputs. 
    # anchor is the target coordinate to integrate all feature maps.
    def feature_fusion(self, feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, tgt_depth, anchor):
        with torch.no_grad():

            B_,C_,H_,W_ = feat_.shape
            feat_rgb_ = feat_[:B_//3,...]
            feat_nir_ = feat_[B_//3:2*B_//3,...]
            feat_thr_ = feat_[2*B_//3:, ...]

            if anchor == 'RGB': 
                prj_nir, valid_mask_nir = self.feature_projection(feat_nir_, intrinsics_nir, intrinsics_rgb, extrinsics['RGB2NIR'], tgt_depth)
                prj_thr, valid_mask_thr = self.feature_projection(feat_thr_, intrinsics_thr, intrinsics_rgb, extrinsics['RGB2THR'], tgt_depth)

                align_rgb = feat_rgb_
                align_nir = prj_nir*valid_mask_nir.unsqueeze(1)
                align_thr = prj_thr*valid_mask_thr.unsqueeze(1)

            elif anchor == 'NIR':
                prj_rgb, valid_mask_rgb = self.feature_projection(feat_rgb_, intrinsics_rgb, intrinsics_nir, extrinsics['NIR2RGB'], tgt_depth)
                prj_thr, valid_mask_thr = self.feature_projection(feat_thr_, intrinsics_thr, intrinsics_nir, extrinsics['NIR2THR'], tgt_depth)

                align_rgb = prj_rgb*valid_mask_rgb.unsqueeze(1)
                align_nir = feat_nir_
                align_thr = prj_thr*valid_mask_thr.unsqueeze(1)

            elif anchor == 'THR':
                prj_rgb, valid_mask_rgb = self.feature_projection(feat_rgb_, intrinsics_rgb, intrinsics_thr, extrinsics['THR2RGB'], tgt_depth)
                prj_nir, valid_mask_nir = self.feature_projection(feat_nir_, intrinsics_nir, intrinsics_thr, extrinsics['THR2NIR'], tgt_depth)

                align_rgb = prj_rgb*valid_mask_rgb.unsqueeze(1)
                align_nir = prj_nir*valid_mask_nir.unsqueeze(1)
                align_thr = feat_thr_

            shared_rgb   = align_rgb[:,:C_//2,...]
            specific_rgb = align_rgb[:,C_//2:,:,...]

            shared_nir   = align_nir[:,:C_//2,...]
            specific_nir = align_nir[:,C_//2:,:,...]

            shared_thr   = align_thr[:,:C_//2,...]
            specific_thr = align_thr[:,C_//2:,:,...]

            # prior knowledge : generally thermal feature is more robust than others
            # thus, cosince similarity between two features remains reliable feature only.
            score_rgb = self.loss_cosine(shared_thr, shared_rgb).unsqueeze(1)
            score_nir = self.loss_cosine(shared_thr, shared_nir).unsqueeze(1)
            score_thr = torch.ones_like(score_nir)

            # naive fusion
            # fused_feat = torch.cat((feat_thr_, prj_nir2thr, prj_rgb2thr), dim=1)

            # selective fusion
            fused_share = (score_thr*shared_thr + score_rgb*shared_rgb + score_nir*shared_nir) / (score_thr+score_rgb+score_nir)
            fused_feat = torch.cat((fused_share, score_thr*specific_thr, score_nir*specific_nir, score_rgb*specific_rgb), dim=1)
            return fused_feat

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        w1 = self.opt.model.sup_depth_weight
        w2 = self.opt.model.smooth_weight
        w3 = self.opt.model.consistency_weight
        w4 = self.opt.model.consist_balance_weight
        w5 = self.opt.model.geometry_weight

        # order : rgb, nir, thr 
        tgt_img_input1, tgt_img_vis1, tgt_gt_depth1, intrinsics_rgb = batch["rgb"]["tgt_image"], batch["rgb"]["tgt_image_eh"],\
                                                                      batch["rgb"]["tgt_depth_gt"], batch["rgb"]["intrinsics"] 
        tgt_img_input2, tgt_img_vis2, tgt_gt_depth2, intrinsics_nir = batch["nir"]["tgt_image"], batch["nir"]["tgt_image_eh"],\
                                                                      batch["nir"]["tgt_depth_gt"], batch["nir"]["intrinsics"] 
        tgt_img_input3, tgt_img_vis3, tgt_gt_depth3, intrinsics_thr = batch["thr"]["tgt_image"], batch["thr"]["tgt_image_eh"],\
                                                                      batch["thr"]["tgt_depth_gt"], batch["thr"]["intrinsics"] 
        extrinsics = batch["extrinsics"]

        tgt_img = torch.cat([tgt_img_input1, tgt_img_input2.repeat_interleave(3, axis=1), tgt_img_input3.repeat_interleave(3, axis=1)], dim=0)
        tgt_img_vis = torch.cat([tgt_img_vis1, tgt_img_vis2.repeat_interleave(3, axis=1), tgt_img_vis3.repeat_interleave(3, axis=1)], dim=0)
        tgt_gt_depth = torch.cat([tgt_gt_depth1, tgt_gt_depth2, tgt_gt_depth3], dim=0)

        # 1. Modality-wise inference 
        pred_depths, feats = self.forward(tgt_img)
        if 'midas' in self.opt.model.mode:
            pred_depths = pred_depths.unsqueeze(1) #B1HW, B: batch x 3 (rgb, nir, thr)

        B_, _,  _,_ = pred_depths.shape
        pred_depth1 = pred_depths[:B_//3,...]           
        pred_depth2 = pred_depths[B_//3:2*B_//3,...]   
        pred_depth3 = pred_depths[2*B_//3:, ...]   

        # initialize for logger
        loss_cont = torch.tensor(0).float().to(tgt_img.device)
        loss_geo  = torch.zeros_like(loss_cont)

        # Select Stage: Multi-spectral fused depth via selective feature fusion
        if self.flag_fuse :
            # 2.1 feature projection & Merge
            merged_feats = []
            for idx, feat_ in zip(range(0, len(feats)), feats) :
                fused_feat_rgb = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, pred_depth1, anchor='RGB')
                fused_feat_nir = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, pred_depth2, anchor='NIR')
                fused_feat_thr = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, pred_depth3, anchor='THR')
                merged_feats.append(torch.cat([fused_feat_rgb, fused_feat_nir, fused_feat_thr], dim=0))
    
            # 2.2 feagure aggregation layer & forward
            agg_feats = self.fusion_layer(merged_feats)
            pred_depths = self.depth_net.forward_depth(agg_feats)
            if 'midas' in self.opt.model.mode:
                pred_depths = pred_depths.unsqueeze(1)

            B_, _,  _,_ = pred_depths.shape
            pred_depth1 = pred_depths[:B_//3,...] # RGB camera centered depth          
            pred_depth2 = pred_depths[B_//3:2*B_//3,...] # NIR camera centered depth   
            pred_depth3 = pred_depths[2*B_//3:, ...] # Thermal camera centered depth   

        # Align Stage: Spectral-wise depth via contrastive learning
        else : 
            # 1.1. Global & Dence Feature Alignment 
            global_consist_loss = torch.zeros_like(loss_geo)
            local_consist_loss = torch.zeros_like(loss_geo)
            for idx, feat_ in zip(range(0, len(feats)), feats) :
                if str(idx) in self.opt.model.consis_idx:

                    B_,C_,H_,W_ = feat_.shape
                    feat_rgb_ = feat_[:B_//3,...]           
                    feat_nir_ = feat_[B_//3:2*B_//3,...]   
                    feat_thr_ = feat_[2*B_//3:, ...]   

                    # 2-1. Global contrastive loss
                    # We induce each former half channels are shared between modalities, remaning half channels are modality-specific.
                    query = self.global_pool(feat_rgb_[:,:C_//2,...].detach()).squeeze()
                    positive_nir = self.global_pool(feat_nir_[:,:C_//2,...]).squeeze()
                    positive_thr = self.global_pool(feat_thr_[:,:C_//2,...]).squeeze()

                    negative_rgb = self.global_pool(feat_rgb_[:,C_//2:,...]).squeeze()
                    negative_nir = self.global_pool(feat_nir_[:,C_//2:,...]).squeeze()
                    negative_thr = self.global_pool(feat_thr_[:,C_//2:,...]).squeeze()

                    query = query.repeat_interleave(2, axis=0)
                    positive = torch.cat((positive_nir, positive_thr), axis=0)
                    negative = torch.cat((negative_rgb, negative_nir, negative_thr), axis=0)
                    global_consist_loss += self.loss_cont_g(query, positive, negative)

                    # 2-2. Local contrastive loss
                    # project feature map (RGB->THR, RGB->NIR)
                    prj_rgb, valid_mask_rgb = self.feature_projection(feat_rgb_, intrinsics_rgb, intrinsics_thr, extrinsics['THR2RGB'], pred_depth3)
                    prj_nir, valid_mask_nir = self.feature_projection(feat_nir_, intrinsics_nir, intrinsics_thr, extrinsics['THR2NIR'], pred_depth3)

                    # 2-2.  feature seperation
                    shared_rgb   = prj_rgb[:,:C_//2,...]*valid_mask_rgb.unsqueeze(1).detach()
                    specific_rgb = prj_rgb[:,C_//2:,:,...]*valid_mask_rgb.unsqueeze(1)

                    shared_nir   = prj_nir[:,:C_//2,...]*valid_mask_nir.unsqueeze(1)
                    specific_nir = prj_nir[:,C_//2:,:,...]*valid_mask_nir.unsqueeze(1)

                    shared_thr   = feat_thr_[:,:C_//2,...]
                    specific_thr = feat_thr_[:,C_//2:,:,...]

                    if (valid_mask_rgb.sum()>(H_*W_)/10) & (valid_mask_nir.sum()>(H_*W_)/10) :
                        query = shared_rgb.repeat_interleave(2, axis=0)
                        positive = torch.cat((shared_nir, shared_thr),axis=0)
                        negative = torch.cat((specific_nir, specific_thr),axis=0)
                        local_consist_loss += self.loss_cont_l(query, positive, negative)

                loss_cont = ((1-w4)*global_consist_loss + w4*local_consist_loss)

        # supervised loss
        loss_sup1 = self.loss_sup(pred_depth1.squeeze(), tgt_gt_depth1)
        loss_sup2 = self.loss_sup(pred_depth2.squeeze(), tgt_gt_depth2)
        loss_sup3 = self.loss_sup(pred_depth3.squeeze(), tgt_gt_depth3)
        loss_sup = loss_sup1+loss_sup2+loss_sup3

        # depth smoothness loss
        loss_sm  = self.loss_smooth(pred_depth1, tgt_img_vis1)
        loss_sm  += self.loss_smooth(pred_depth2, tgt_img_vis2)
        loss_sm  += self.loss_smooth(pred_depth3, tgt_img_vis3)

        # geometric consistency loss (for select stage), follows more accurate depth map
        if self.flag_fuse :  
            if((loss_sup3 < loss_sup1)&(loss_sup3 < loss_sup2)): # THR
                loss_geo += self.geometric_consistency_loss(pred_depth1, intrinsics_rgb, intrinsics_thr, extrinsics['THR2RGB'], pred_depth3)
                loss_geo += self.geometric_consistency_loss(pred_depth2, intrinsics_nir, intrinsics_thr, extrinsics['THR2NIR'], pred_depth3)
            elif((loss_sup1 < loss_sup2)&(loss_sup1 < loss_sup3)): # RGB
                loss_geo += self.geometric_consistency_loss(pred_depth2, intrinsics_nir, intrinsics_rgb, extrinsics['RGB2NIR'], pred_depth1)
                loss_geo += self.geometric_consistency_loss(pred_depth3, intrinsics_thr, intrinsics_rgb, extrinsics['RGB2THR'], pred_depth1)
            else: # NIR
                loss_geo += self.geometric_consistency_loss(pred_depth1, intrinsics_rgb, intrinsics_nir, extrinsics['NIR2RGB'], pred_depth2)
                loss_geo += self.geometric_consistency_loss(pred_depth3, intrinsics_thr, intrinsics_nir, extrinsics['NIR2THR'], pred_depth2)

        # total loss
        loss = w1*loss_sup + w2*loss_sm + w3*loss_cont + w5*loss_geo 

        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        if(self.lrscheduler_batch):
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

        # create logs
        self.log('train/total_loss', loss)
        self.log('train/sup_loss', loss_sup)
        self.log('train/smooth_loss', loss_sm)
        self.log('train/consis_loss', loss_cont)
        self.log('train/geo_loss', loss_geo)
        return loss

    def training_epoch_end(self, outputs):
        if(self.lrscheduler_epoch):
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        tgt_img_input1, tgt_img_vis1, gt_depth1, intrinsics_rgb = batch["rgb"]["tgt_image"], batch["rgb"]["tgt_image_eh"],\
                                                                  batch["rgb"]["tgt_depth_gt"], batch["rgb"]["intrinsics"] 
        tgt_img_input2, tgt_img_vis2, gt_depth2, intrinsics_nir = batch["nir"]["tgt_image"], batch["nir"]["tgt_image_eh"],\
                                                                  batch["nir"]["tgt_depth_gt"], batch["nir"]["intrinsics"] 
        tgt_img_input3, tgt_img_vis3, gt_depth3, intrinsics_thr = batch["thr"]["tgt_image"], batch["thr"]["tgt_image_eh"],\
                                                                  batch["thr"]["tgt_depth_gt"], batch["thr"]["intrinsics"] 
        extrinsics = batch["extrinsics"]

        # Spectral-wise depth
        tgt_img = torch.cat([tgt_img_input1, tgt_img_input2.repeat_interleave(3, axis=1), tgt_img_input3.repeat_interleave(3, axis=1)], dim=0)
        gt_depth = torch.cat([gt_depth1, gt_depth2, gt_depth3], dim=0)
        tgt_depth, feats = self.forward(tgt_img)
        if 'midas' in self.opt.model.mode:
            tgt_depth = tgt_depth.unsqueeze(1)

        B_, _, _,_ = tgt_depth.shape
        tgt_depth1 = tgt_depth[:B_//3,...]           
        tgt_depth2 = tgt_depth[B_//3:2*B_//3,...]   
        tgt_depth3 = tgt_depth[2*B_//3:, ...]   

        # Multi-spectral depth 
        if self.flag_fuse :
            merged_feats = []
            for idx, feat_ in zip(range(0, len(feats)), feats) :
                fused_feat_rgb = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, tgt_depth1, anchor='RGB')
                fused_feat_nir = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, tgt_depth2, anchor='NIR')
                fused_feat_thr = self.feature_fusion(feat_, intrinsics_rgb, intrinsics_nir, intrinsics_thr, extrinsics, tgt_depth3, anchor='THR')

                merged_feats.append(torch.cat([fused_feat_rgb, fused_feat_nir, fused_feat_thr], dim=0))
    
            # 2.3 inference 
            aggregated_feat = self.fusion_layer(merged_feats)
            tgt_depth = self.depth_net.forward_depth(aggregated_feat)
            if 'midas' in self.opt.model.mode:
                tgt_depth = tgt_depth.unsqueeze(1)

            B_, _, _,_ = tgt_depth.shape
            tgt_depth1 = tgt_depth[:B_//3,...]           
            tgt_depth2 = tgt_depth[B_//3:2*B_//3,...]   
            tgt_depth3 = tgt_depth[2*B_//3:, ...]   

        errs = compute_depth_errors(gt_depth, tgt_depth)
        errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        # plot
        if batch_idx < 3:
            if tgt_img_vis1[0].nelement() != tgt_depth1[0].nelement():
                C,H,W = tgt_img_vis1[0].shape
                tgt_depth1 = torch.nn.functional.interpolate(tgt_depth1, [H, W], mode='nearest')
                tgt_depth2 = torch.nn.functional.interpolate(tgt_depth2, [H, W], mode='nearest')
                tgt_depth3 = torch.nn.functional.interpolate(tgt_depth3, [H, W], mode='nearest')

            vis_img = visualize_image(tgt_img_vis1[0])  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth1[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_depth_s1_{}'.format(batch_idx), stack, self.current_epoch)

            vis_img = visualize_image(tgt_img_vis2[0])  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth2[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_depth_s2_{}'.format(batch_idx), stack, self.current_epoch)

            vis_img = visualize_image(tgt_img_vis3[0])  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth3[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_depth_s3_{}'.format(batch_idx), stack, self.current_epoch)

        return errs

    def validation_epoch_end(self, outputs):
        mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
        mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
        mean_a1 = np.array([x['a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['a3'] for x in outputs]).mean()

        self.log('val_loss', mean_rel, prog_bar=True, sync_dist=True)
        self.log('val/abs_diff', mean_diff, sync_dist=True)
        self.log('val/abs_rel', mean_rel, sync_dist=True)
        self.log('val/a1', mean_a1, on_epoch=True, sync_dist=True)
        self.log('val/a2', mean_a2, on_epoch=True, sync_dist=True)
        self.log('val/a3', mean_a3, on_epoch=True, sync_dist=True)

