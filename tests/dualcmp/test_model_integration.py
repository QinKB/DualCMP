"""Environment-gated real frozen-window learning smoke."""
import os
from pathlib import Path
import pytest
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.tools.dualcmp.protocol import DETECTOR_COMPRESSION

@pytest.mark.integration
def test_frozen_t3_backward():
    root, weights = os.environ["V2V4REAL_ROOT"], os.environ["DUALCMP_DETECTOR_WEIGHTS"]
    h=load_yaml("opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml"); h["root_dir"]=str(Path(root)/"test"); h["fusion"]={"core_method":"DualCMPSequenceDataset","args":{"stage":"tracking","tracking_frames":3,"runtime_only":False,"runtime_forbid_gt":True,"receiver_ids":[0],"frame_period_s":.1,"max_actors":128,"max_cav":2,"window":[{"scene_id":"testoutput_CAV_data_2022-03-17-11-02-23_2","receiver_id":0,"start_frame":237,"end_frame":260}]}}
    d=DualCMPSequenceDataset(h,train=True); batch=d.collate_batch_train([d[0]]); h["model"]["args"]["compression"]=DETECTOR_COMPRESSION
    model=DualCmpClipModel(h["model"]["args"],d.anchor_box,h["preprocess"]["cav_lidar_range"]); model.detector.load_state_dict(torch.load(weights,map_location="cpu"),strict=True)
    outputs,_=model(batch["runtime_frames"]); [x["fused_bev"].retain_grad() for x in outputs]; [x["roi_feature"].retain_grad() for x in outputs]
    loss=DualCmpLoss({"detector":h["loss"]["args"]})(outputs,batch["target_frames"]); loss["total_loss"].backward()
    values={key:float(loss[key].detach()) for key in ("L_det","L_exist","L_box","L_velocity")}; grads={"track_update":float(model.track_update[0].weight.grad.norm()),"roi":float(sum(x["roi_feature"].grad.norm() for x in outputs)),"fused":float(sum(x["fused_bev"].grad.norm() for x in outputs)),"backbone":float(model.detector.backbone.blocks[0][1].weight.grad.norm())}; print({"loss":values,"match":loss["match_count"],"grad":grads})
    assert loss["match_count"]>0 and all(torch.isfinite(loss[key]) for key in ("L_det","L_exist","L_box","L_velocity")) and all(value>0 for value in grads.values())
