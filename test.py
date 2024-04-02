import os, sys 
import random
import numpy as np
import torch
import argparse
import cv2
import time
import datetime

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = data_dir+'/'+args.logname
    scene = Scene_mica(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background, device = args.device)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_save_path = os.path.join(logdir, 'test.avi')
    out = cv2.VideoWriter(vid_save_path, fourcc, 25, (args.image_res*2, args.image_res), True)

    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    for iteration in range(len(viewpoint)):
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        save_image = np.zeros((args.image_res, args.image_res*2, 3))
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        save_image[:, :args.image_res, :] = gt_image_np
        save_image[:, args.image_res:, :] = image_np
        save_image = save_image.astype(np.uint8)
        save_image = save_image[:,:,[2,1,0]]

        out.write(save_image)
    out.release()
    
    
   
        

           