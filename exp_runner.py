import os
import time
import json
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import sys


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        # self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device =="cuda" and torch_version < "2.":
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
        
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        
        def serialize(obj, possible_keys=['data_dir', 
                                          'render_cameras_name',
                                          'object_cameras_name',
                                          'n_images',
                                          'image_indices']):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()  # Convert PyTorch tensor to a list
            if isinstance(obj, list):
                return [serialize(item) for item in obj]  # Recursively serialize lists
            if hasattr(obj, '__dict__'):
                return {key: serialize(value) for key, value in obj.__dict__.items() if key in possible_keys}  # Convert objects recursively
            return obj  # Keep other data as is
        dataset_path = os.path.join(self.base_exp_dir, "dataset_info.json")
        dataset_serializable = serialize(self.dataset)
        with open(dataset_path, "w") as json_file:
            json.dump(dataset_serializable, json_file, indent=4)
            
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
            
        # Maps parameters   
        self.guided_sampling = self.conf.get_bool('sampling.guided_sampling', default=False)
        self.val_ps_freq = self.conf.get_int('sampling.val_ps_freq', default=5000)
        if self.guided_sampling:
            self.Rx = self.conf.get_int('sampling.resX', default=32)
            self.Ry = self.conf.get_int('sampling.resY', default=32)
            self.Rz = self.conf.get_int('sampling.resZ', default=32)
            self.F = self.conf.get_int('sampling.factF', default=2)
            
            self.world_mats, self.scale_mats = self.dataset.get_world_scale_maps()
            self.H, self.W = self.dataset.get_image_size()
            self.maps = self.compute_maps()
            
        if self.device == "cuda":
            if torch.cuda.device_count() > 1:
                self.nerf_outside = torch.nn.DataParallel(self.nerf_outside)
                self.sdf_network = torch.nn.DataParallel(self.sdf_network)
                self.deviation_network = torch.nn.DataParallel(self.deviation_network)
                self.color_network = torch.nn.DataParallel(self.color_network)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            if self.guided_sampling:
                data1 = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size // 2)
                data2 = self.dataset.gen_guided_rays_at(image_perm[(self.iter_step + 1) % len(image_perm)], self.maps[image_perm[(self.iter_step + 1) % len(image_perm)]], self.batch_size // 2, self.Rx, self.Ry)
                data = torch.concatenate([data1, data2], axis=0)
            
            else:
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            
            rays_o, rays_d, true_rgb, mask = data[:, :3].to(self.device), \
                                 data[:, 3:6].to(self.device), \
                                 data[:, 6:9].to(self.device), \
                                 data[:, 9:10].to(self.device)

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()
            
            if self.guided_sampling & (self.iter_step % self.val_ps_freq == 0):
                self.maps = self.compute_maps()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()
    
    ### Added a function to compute the guided sampling maps (see notebook/report for more information)
    def compute_map(self, world_mat, scale_mat, W, H, s=1):

        # Define geometric transformation
        def h(x, P) :
            x = torch.concat([x, torch.ones((x.shape[0], 1))], axis=1)
            return P @ x.T

        def g(x_hat) :
            x_hat [:2, :] = x_hat[:2, :] / x_hat[2, :]
            return x_hat

        def f(x, P) :
            return g(h(x, P))
        
        # Define logistic function
        def Phi(o, s) :
            return s * torch.exp(-s * o) / (1 + torch.exp(-s * o))
        
        # Define projection matrix P
        P = torch.Tensor(world_mat @ scale_mat)[:3, :4].to(self.device)

        limX = 1 # Object are more or less in the unit sphere, check the IDR github page https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md
        
        # Compute the grid of points in the world system
        grid_Xx = torch.linspace(-limX, limX, self.Rx * self.F)
        grid_Xy = torch.linspace(-limX, limX, self.Ry * self.F)
        grid_Xz = torch.linspace(-limX, limX, self.Rz * self.F)

        grid_Xx, grid_Xy, grid_Xz = torch.meshgrid(grid_Xx, grid_Xy, grid_Xz, indexing='xy')
        grid_Xx = grid_Xx.reshape(-1, 1)
        grid_Xy = grid_Xy.reshape(-1, 1)
        grid_Xz = grid_Xz.reshape(-1, 1)

        # Compute the grid of points in the camera system
        grid_Ux = torch.linspace(0, W, self.Rx)
        grid_Uy = torch.linspace(0, H, self.Ry)

        distancex = 0.5 * (grid_Ux[1] - grid_Ux[0])
        distancey = 0.5 * (grid_Uy[1] - grid_Uy[0])

        grid_Ux, grid_Uy = torch.meshgrid(grid_Ux, grid_Uy, indexing='xy')
        grid_Ux = grid_Ux.reshape(-1, 1)
        grid_Uy = grid_Uy.reshape(-1, 1)
        
        # Compute the grid of points in the world-to-camera system
        points_world = torch.cat([grid_Xx, grid_Xy, grid_Xz], dim=1)
        grid_fXx, grid_fXy, grid_fXz = f(points_world, P)

        # Compute the PDF of world points (points --> SDF --> PDF)
        with torch.no_grad():
            points_world = points_world.to(self.device)
            sdf_values = self.sdf_network.sdf(points_world)
            pdf_values = Phi(sdf_values, s)
            pdf_values = pdf_values.T
        
        # Compute the projected SDF in the camera system
        Ux_min = grid_Ux - distancex
        Ux_max = grid_Ux + distancex
        Uy_min = grid_Uy - distancey
        Uy_max = grid_Uy + distancey

        UXx_intersect = (grid_fXx < Ux_max) & (grid_fXx > Ux_min) & (grid_fXx > 0)
        UXy_intersect = (grid_fXy < Uy_max) & (grid_fXy > Uy_min) & (grid_fXy > 0)

        mask = UXx_intersect & UXy_intersect
        
        probs = torch.sum(pdf_values * mask, axis=1, keepdims=True)
        
        torch.cuda.empty_cache()
        
        return probs
    
    def compute_maps(self):
        print('Computing guided sampling maps...')
        maps = []
        for i in range(len(self.world_mats)):
            maps.append(self.compute_map(self.world_mats[i], self.scale_mats[i], self.H, self.W))
            
            if i % 10 == 0:
                print(f'Computing map {i}/{len(self.world_mats)}')
                try:
                    print(f'Norm diff: {torch.norm(self.maps[i] - maps[i]):.4f}')
                except:
                    pass
        
        return maps


if __name__ == '__main__':
    print('Hello Wooden')
    
    torch_version = torch.__version__.split("+")[0]
    print(f"torch version: {torch_version}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on : {device}")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    # Filter out Jupyter-specific arguments
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]

    # Filter out Jupyter-specific arguments
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/project_DTU.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='scan24')

    args = parser.parse_args()
    
    if torch_version < "2." :
        if torch.cuda.is_available() :
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print(f"Using {torch.cuda.device_count()} GPUs")
            if torch.cuda.device_count()==1 :
                print(f"Using GPU {args.gpu}")
                torch.cuda.set_device(args.gpu)
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
    if torch_version >= "2." :
        # Set default data type
        torch.set_default_dtype(torch.float32)
        # Set default device
        torch.set_default_device('cuda')
        
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)