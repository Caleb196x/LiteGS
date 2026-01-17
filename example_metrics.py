from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
import matplotlib.pyplot as plt
import json

import litegs
import litegs.config
import litegs.utils
import shutil


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--analyze_per_image", default=False, action="store_true")   # 开启 per-image 分析功能
    parser.add_argument("--worst_percent", type=float, default=0.05)  # 最差比例，默认5%
    args = parser.parse_args(sys.argv[1:])
    
    lp=litegs.arguments.ModelParams.extract(args)
    op=litegs.arguments.OptimizationParams.extract(args)
    pp=litegs.arguments.PipelineParams.extract(args)
    dp=litegs.arguments.DensifyParams.extract(args)

    cameras_info:dict[int,litegs.data.CameraInfo]=None
    camera_frames:list[litegs.data.ImageFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution


    if args.save_image:
        try:
            shutil.rmtree(os.path.join(lp.model_path,"Trainingset"))
            shutil.rmtree(os.path.join(lp.model_path,"Testset"))
        except:
            pass
        os.makedirs(os.path.join(lp.model_path,"Trainingset"),exist_ok=True)
        os.makedirs(os.path.join(lp.model_path,"Testset"),exist_ok=True)

    if args.analyze_per_image:
        try:
            shutil.rmtree(os.path.join(lp.model_path,"WorstViews"))
        except:
            pass
        os.makedirs(os.path.join(lp.model_path,"WorstViews"),exist_ok=True)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        if os.path.exists(os.path.join(lp.source_path,"train_test_split.json")):
            with open(os.path.join(lp.source_path,"train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames=[c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames=[c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
        trainingset=litegs.data.CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
        testset=litegs.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    else:
        trainingset=litegs.data.CameraFrameDataset(cameras_info,camera_frames,lp.resolution,pp.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #model
    xyz,scale,rot,sh_0,sh_rest,opacity=litegs.io_manager.load_ply(os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply"),lp.sh_degree)
    xyz=torch.Tensor(xyz).cuda()
    scale=torch.Tensor(scale).cuda()
    rot=torch.Tensor(rot).cuda()
    sh_0=torch.Tensor(sh_0).cuda()
    sh_rest=torch.Tensor(sh_rest).cuda()
    opacity=torch.Tensor(opacity).cuda()
    cluster_origin=None
    cluster_extend=None
    if pp.cluster_size>0:
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.point.spatial_refine(False,None,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        cluster_origin,cluster_extend=litegs.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    if op.learnable_viewproj:
        noise_extr=torch.cat([frame.extr_params[None,:] for frame in trainingset.frames])
        noise_intr=torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0)
        denoised_training_extr,denoised_training_intr=torch.load(os.path.join(lp.model_path,"point_cloud","finish","viewproj.pth"))

    #metrics
    ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
    lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    #iter
    if lp.eval:
        loaders={"Trainingset":train_loader,"Testset":test_loader}
    else:
        loaders={"Trainingset":train_loader}

    with torch.no_grad():
        for loader_name,loader in loaders.items():
            ssim_list=[]
            psnr_list=[]
            lpips_list=[]
            if args.analyze_per_image:
                metrics_records=[]  # 存储 {name, psnr, ssim, lpips, img, gt} 的列表
            for index,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(loader):
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                idx=idx.cuda()
                if op.learnable_viewproj:
                    if loader_name=="Trainingset":
                        #fix view matrix
                        extr=denoised_training_extr[idx]
                        intr=denoised_training_intr
                    else:
                        nearest_idx=(extr-denoised_training_extr).abs().sum(dim=1).argmin()
                        delta=denoised_training_extr[nearest_idx]-noise_extr[nearest_idx]
                        extr=extr+delta
                    view_matrix,proj_matrix,viewproj_matrix,frustumplane=litegs.utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=litegs.render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,lp.sh_degree)
                img,transmitance,depth,normal,primitive_visible=litegs.render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                            lp.sh_degree,gt_image.shape[2:],pp)
                psnr_value=psnr_metrics(img,gt_image)
                ssim_value=ssim_metrics(img,gt_image)
                lpips_value=lpip_metrics(img,gt_image)
                ssim_list.append(ssim_value.unsqueeze(0))
                psnr_list.append(psnr_value.unsqueeze(0))
                lpips_list.append(lpips_value.unsqueeze(0))
                if args.analyze_per_image:
                    frame_name=loader.dataset.frames[int(idx.item())].name
                    metrics_records.append({
                        'name': frame_name,
                        'psnr': float(psnr_value),
                        'ssim': float(ssim_value),
                        'lpips': float(lpips_value),
                        'img': img.detach().cpu()[0].permute(1,2,0).numpy(),
                        'gt': gt_image.detach().cpu()[0].permute(1,2,0).numpy()
                    })
                if loader_name=="Testset" and args.save_image:
                    plt.imsave(os.path.join(lp.model_path,loader_name,"{}-{:.2f}-rd.png".format(index,float(psnr_value))),img.detach().cpu()[0].permute(1,2,0).numpy())
                    plt.imsave(os.path.join(lp.model_path,loader_name,"{}-{:.2f}-gt.png".format(index,float(psnr_value))),gt_image.detach().cpu()[0].permute(1,2,0).numpy())
            ssim_mean=torch.concat(ssim_list,dim=0).mean()
            psnr_mean=torch.concat(psnr_list,dim=0).mean()
            lpips_mean=torch.concat(lpips_list,dim=0).mean()

            print("  Scene:{0}".format(lp.model_path+" "+loader_name))
            print("  SSIM : {:>12.7f}".format(float(ssim_mean)))
            print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
            print("  LPIPS: {:>12.7f}".format(float(lpips_mean)))
            print("")

            if args.analyze_per_image:
                # 计算综合评分（归一化后加权）
                # 公式: score = normalized_psnr * 0.4 + normalized_ssim * 0.3 - normalized_lpips * 0.3
                # 分数越低越差
                psnr_values = [r['psnr'] for r in metrics_records]
                ssim_values = [r['ssim'] for r in metrics_records]
                lpips_values = [r['lpips'] for r in metrics_records]

                psnr_min, psnr_max = min(psnr_values), max(psnr_values)
                ssim_min, ssim_max = min(ssim_values), max(ssim_values)
                lpips_min, lpips_max = min(lpips_values), max(lpips_values)

                for record in metrics_records:
                    norm_psnr = (record['psnr'] - psnr_min) / (psnr_max - psnr_min + 1e-8)
                    norm_ssim = (record['ssim'] - ssim_min) / (ssim_max - ssim_min + 1e-8)
                    norm_lpips = (record['lpips'] - lpips_min) / (lpips_max - lpips_min + 1e-8)
                    record['score'] = norm_psnr * 0.4 + norm_ssim * 0.3 - norm_lpips * 0.3

                # 按综合评分排序（升序，最差在前）
                sorted_records = sorted(metrics_records, key=lambda x: x['score'])
                worst_count = max(1, int(len(sorted_records) * args.worst_percent))
                worst_records = sorted_records[:worst_count]

                # 输出 per-image 指标表
                print("  Per-image metrics:")
                print("  {:<30} {:>10} {:>10} {:>10} {:>10}".format("Name", "PSNR", "SSIM", "LPIPS", "Score"))
                print("  " + "-" * 74)
                for record in sorted_records:
                    print("  {:<30} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                        record['name'], record['psnr'], record['ssim'], record['lpips'], record['score']))

                # 输出最差视角并保存图片
                print("")
                print("  Worst {:.0f}% views ({} images):".format(args.worst_percent*100, worst_count))
                for record in worst_records:
                    print("    {} - PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}, Score: {:.4f}".format(
                        record['name'], record['psnr'], record['ssim'], record['lpips'], record['score']))
                    # 保存渲染结果和GT图片
                    plt.imsave(os.path.join(lp.model_path, "WorstViews", "{}_render.png".format(record['name'])),
                               record['img'])
                    plt.imsave(os.path.join(lp.model_path, "WorstViews", "{}_gt.png".format(record['name'])),
                               record['gt'])
                print("")
