import torch
from models import HarMABase, load_pretrained_harma
import torch.nn.functional as F

import torch
# from torchinfo import summary
from PIL import Image
import open_clip
# from inference_tool import get_preprocess
from open_clip import tokenizer

# clip, _, _ = open_clip.create_model_and_transforms("ViT-B/32")
# checkpoint = torch.load(ckpt_path, map_location="cpu")
# msg = clip.load_state_dict(checkpoint, strict=False)
# print("Missing keys: ", msg.missing_keys)
# print("Unexpected keys: ", msg.unexpected_keys)


class HarMA(HarMABase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.create_and_load_pretrained(config)
        self.align_before = False

    def create_and_load_pretrained(self, config):
        if self.config['model'] == 'geo': 
            self.clip, _ ,_ = open_clip.create_model_and_transforms("ViT-B/32",pretrained='openai')
            ckpt_path = "./pretrain/RS5M_ViT-B-32_RET-2.pt"
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            msg = self.clip.load_state_dict(checkpoint, strict=False)
        else:
            self.clip, _, _ = open_clip.create_model_and_transforms("ViT-B/32")

    def get_vis_emb(self, image, idx=None, label=None):
        if self.config['is_harma']:
            if self.align_before:
                img_emb,feas_vis = self.clip.encode_image(image,normalize=True)
                return img_emb,feas_vis
            else:
                img_emb = self.clip.encode_image(image,normalize=True)
            return img_emb
        
    def get_txt_emb(self, text_ids, idx=None, label=None):
        if self.config['is_harma']:
            if self.align_before:
                txt_emb,feas_txt = self.clip.encode_text(text_ids,normalize=True)
                return txt_emb,feas_txt
            else:
                txt_emb = self.clip.encode_text(text_ids,normalize=True)
            return txt_emb
        

    def forward(self, image, text_ids, idx=None, label=None):
        ## Baseline(Swin-T+Bert-B)
        if self.config['is_harma']:
            if self.align_before:
                img_emb,feas_vis = self.get_vis_emb(image)
                txt_emb,feas_txt = self.get_txt_emb(text_ids)
            else:
                img_emb = self.get_vis_emb(image)
                txt_emb=self.get_txt_emb(text_ids)
            # txt_emb = self.get_text_embeds(text_ids)
        else:
            img_emb= self.get_vision_fusion_embeds(image, self.config)
            txt_emb = self.get_text_fusion_embeds(text_ids, self.config)

        if self.use_affil_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr, loss_affil
        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_before_contr = []
            if self.align_before:
                for i in range(len(feas_vis)):
                    # print("vis",feas_vis[i].shape)
                    loss_contr = self.get_contr_loss(feas_vis[i],feas_txt[i], idx=idx, label=label, config=self.config)
                    loss_before_contr.append(loss_contr)
                total_loss_before = sum(loss_before_contr)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            if self.align_before:
                return loss_contr,loss_triplet,total_loss_before
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            #TODO new loss
            return loss_contr,loss_triplet,_