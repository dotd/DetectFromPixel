#!/usr/bin python3
# -*- coding: utf-8 -*-
import argparse
import json
import multiprocessing as multi
import os
import os.path as osp
import time
import glob
import datetime
from pathlib import Path
import h5py
import numpy as np
import pyrealsense2 as rs
import imageio
from multiprocessing import Pool

from definitions_detect_from_pixel import ROOT_DIR


def camera_pipeline_config(camera_sn, config):
    cfg = rs.config()
    cfg.enable_device(camera_sn)
    cfg.enable_stream(
        rs.stream.depth,
        config["width"],
        config["height"],
        rs.format.z16,
        config["fps"],
    )
    cfg.enable_stream(
        rs.stream.color,
        config["width"],
        config["height"],
        rs.format.rgb8,
        config["fps"],
    )
    return cfg


def post_process_depth(depth_frame):

    # dec_filter = rs.decimation_filter()  # reduces size\resolution

    spat_filter = rs.spatial_filter()
    spat_filter.set_option(rs.option.filter_magnitude, 2)  # 2, 1-5
    spat_filter.set_option(rs.option.filter_smooth_alpha, 0.9)  # 0.5, 0.25-1.0
    spat_filter.set_option(rs.option.filter_smooth_delta, 20)  # 20, 1-50
    spat_filter.set_option(rs.option.holes_fill, 0)  # 0, 0-5

    temp_filter = rs.temporal_filter()
    temp_filter.set_option(rs.option.filter_smooth_alpha, 0.7)  # 0.4, 0-1
    temp_filter.set_option(rs.option.filter_smooth_delta, 20)  # 20, 1-100
    temp_filter.set_option(rs.option.holes_fill, 2)  # 3, 0-8

    hole_filter = rs.hole_filling_filter()
    hole_filter.set_option(rs.option.holes_fill, 1)  # 1, 0-2

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # depth_frame = dec_filter.process(depth_frame)
    disparity = depth_to_disparity.process(depth_frame)
    disparity = spat_filter.process(disparity)
    disparity = temp_filter.process(disparity)
    depth_frame = disparity_to_depth.process(disparity)
    depth_frame = hole_filter.process(depth_frame)
    return depth_frame


def convert_depth(depth_data, depth_scale, clamp_max):
    depth_data = np.multiply(depth_data, depth_scale)  # metric
    depth_data = np.minimum(depth_data, clamp_max)
    depth_data = np.uint8(
        np.multiply(depth_data, 255.0 / (clamp_max * depth_scale)))
    return depth_data


def catch_frames(cam_pipe,
                 output_dir,
                 depth_units,
                 depth_scale,
                 clamp_max,
                 post_proc,
                 ):

    frames = cam_pipe.wait_for_frames()
    cam_timestamp = frames.get_timestamp()
    cpu_timestamp = time.time()

    aligner = rs.align(rs.stream.color)

    rgb = frames.get_color_frame()
    depth = frames.get_depth_frame()

    n = len(os.listdir(output_dir))
    images = list()

    time_signature = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    if rgb:
        n += 1
        filename = "image_{}_{:06}.rgb.jpg".format(time_signature, n)
        images.append(filename)
        rgb_data = np.asanyarray(rgb.get_data())
        imageio.imwrite(osp.join(output_dir, filename), rgb_data)
        # img = Image.fromarray(np.asanyarray(rgb.get_data()))
        # img.save(osp.join(output_dir, filename))

    if depth:

        depth_raw = np.asanyarray(depth.get_data())
        depth_mm = np.multiply(depth_raw, depth_units / 1000.0)

        n += 1
        filename = "image_{}_{:06}.depth.h5".format(time_signature, n)
        images.append(filename)
        with h5py.File(osp.join(output_dir, filename), "w") as f:
            f.create_dataset(
                "depth",
                data=depth_mm,
                compression="gzip",
                # chunks=depth_raw.shape,
                # compression_opts=hdf_compression,
                # shape=(depth_raw.shape[0], depth_raw.shape[1], 1),
            )

        aligned_frames = aligner.process(frames)
        depth = aligned_frames.get_depth_frame()
        depth_array = np.asanyarray(depth.get_data())

        n += 1
        filename = "image_{}_{:06}.depth.jpg".format(time_signature, n)
        images.append(filename)
        depth_metric = convert_depth(depth_array, depth_scale, clamp_max)
        imageio.imwrite(osp.join(output_dir, filename), depth_metric)

        if post_proc:
            n += 1
            filename = "image_{}_{:06}.depth_pp.jpg".format(time_signature, n)
            images.append(filename)

            depth_gt_frame = post_process_depth(depth)
            depth_gt = np.asanyarray(depth_gt_frame.get_data())
            depth_gt_metric = convert_depth(depth_gt, depth_scale, clamp_max)
            imageio.imwrite(osp.join(output_dir, filename), depth_gt_metric)

    metadata = dict(
        camera_time=cam_timestamp,
        cpu_time=cpu_timestamp,
        images=images,
    )
    return metadata


def average_depth_over_time(depth_files):
    for k, d_file in enumerate(depth_files):

        with h5py.File(d_file, 'r') as f:
            depth = np.float32(np.array(f["depth"]))

        if k == 0:
            average_depth = np.zeros_like(depth)
            count_non_zero = np.zeros_like(depth)

        denominator = np.maximum(count_non_zero, 1)
        w_prev = np.divide(count_non_zero, denominator)
        w_new = np.divide(1.0 / denominator)
        average_depth = np.add(
            np.multiply(average_depth, w_prev),
            np.multiply(depth, w_new),
        )
        count_non_zero = np.add(count_non_zero, depth > 0)

    return average_depth


def record(params):

    pid = os.getpid()
    pid_str = "pid_{}".format(pid)

    min_valid = params.get("min_valid_frames")
    cam_type = params.get("cam_type")
    warmup = params.get("warmup_frames_d415") if cam_type == "D415" else params.get("warmup_frames_d435")
    camera_sn = params.get("camera_sn")
    path = params.get("path")

    out = "{}_{}_{}".format(path, cam_type, camera_sn)
    #Path(out).mkdir(parents=True, exist_ok=True)
    try:
        os.makedirs(out)
    except OSError:
        if not os.path.isdir(path):
            raise


    cfg = camera_pipeline_config(camera_sn, params)
    cam_pipe = rs.pipeline()
    print("{} starting pipe for device {}".format(pid_str, camera_sn))
    cam_pipe.start(cfg)

    camera_profile = cam_pipe.get_active_profile()
    device = camera_profile.get_device()

    advanced_mode = rs.rs400_advanced_mode(device)
    if advanced_mode.is_enabled():
        print("{} managed to get camera {} into advnaced mode".format(pid_str, camera_sn))
    else:
        advanced_mode.toggle_advanced_mode(True)
        print("Re-trying to enable advanced mode, 5 sec sleep")
        time.sleep(5)
        device = camera_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(device)
        if not advanced_mode.is_enabled():
            raise AssertionError(
                "Failed to enable advnaced mode for camera {}".format(
                    camera_sn))

    depth_units = params.get("depth_units_um")
    clamp_min = params.get("depth_clamp_min")
    clamp_max = params.get("depth_clamp_max")
    disparity_shift = params.get("disparity_shift")

    depth_table = advanced_mode.get_depth_table()
    depth_table.depthUnits = depth_units
    depth_table.depthClampMin = clamp_min
    depth_table.depthClampMax = clamp_max
    depth_table.disparityShift = disparity_shift
    advanced_mode.set_depth_table(depth_table)

    # seems like auto-exposure is default behavior
    for sen in device.sensors:
        sen.set_option(rs.option.enable_auto_exposure, True)

    depth_scale = device.first_depth_sensor().get_depth_scale()

    depth_stream = camera_profile.get_stream(rs.stream.depth)
    color_stream = camera_profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    extrinsics = depth_stream.get_extrinsics_to(color_stream)

    metadata = dict(
        start_time=time.asctime(),
        depth_scale=depth_scale,
        depth_units=depth_units,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        disparity_shift=disparity_shift,
        color_intrinsics=dict(
            width=color_intrinsics.width,
            height=color_intrinsics.height,
            ppx=color_intrinsics.ppx,
            ppy=color_intrinsics.ppy,
            fx=color_intrinsics.fx,
            fy=color_intrinsics.fy,
            model=str(color_intrinsics.model),
            coeffs=color_intrinsics.coeffs,
        ),
        depth_intrinsics=dict(
            width=depth_intrinsics.width,
            height=depth_intrinsics.height,
            ppx=depth_intrinsics.ppx,
            ppy=depth_intrinsics.ppy,
            fx=depth_intrinsics.fx,
            fy=depth_intrinsics.fy,
            model=str(depth_intrinsics.model),
            coeffs=depth_intrinsics.coeffs,
        ),
        extrinsics_depth_to_color=dict(
            translation=extrinsics.translation,
            rotation=extrinsics.rotation,
        ),
        frames=list(),
        warmup=list(),
    )

    post_proc = params.get("post_proc")

    timeout = params.get("timeout")
    timeout_rest = params.get("timeout_rest")
    timeout_repeats = params.get("timeout_repeats")

    for repeat in range(timeout_repeats):
        print("{} Start record {}".format(pid_str, repeat))
        # doing the record
        t0 = time.time()
        while (time.time() - t0) < timeout:
            meta = catch_frames(
                cam_pipe,
                out,
                depth_units,
                depth_scale,
                clamp_max,
                post_proc,
            )
            if len(metadata["warmup"]) < warmup:
                metadata["warmup"].append(meta)
            else:
                metadata["frames"].append(meta)
        # resting
        print("{} Start sleep {}".format(pid_str, repeat))
        if repeat < timeout_repeats - 1:
            time.sleep(timeout_rest)

    cam_pipe.stop()
    print("{} stopped pipe for device {}".format(pid_str, camera_sn))

    with open(osp.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("{} finished recording device {}".format(pid_str, camera_sn))

    rgb_template = osp.join(out, "*.rgb.*")
    n_images = len(glob.glob(rgb_template))
    if n_images < (min_valid + warmup):
        raise AssertionError("{} saved only {} images (required: {})".format(
            pid_str, n_images, min_valid + warmup))
    else:
        print("{} saved {} images".format(pid_str, n_images))


def main(args):
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < args.n_cameras:
        raise RuntimeError("expecting {} RealSense cameras, but only detected {}".format(args.n_cameras, len(devices)))

    path = "{}/data/{}/".format(ROOT_DIR, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    """
    for device in devices:
        params = args.__dict__.copy()
        params["path"] = path
        params["cam_name"] = device.get_info(rs.camera_info.name)
        params["cam_type"] = params["cam_name"].split(" ")[-1]
        params["camera_sn"] = device.get_info(rs.camera_info.serial_number)
        record(params)
    """

    params_pool = list()
    for device in devices:
        params = args.__dict__.copy()
        params["path"] = path
        params["cam_name"] = device.get_info(rs.camera_info.name)
        params["cam_type"] = params["cam_name"].split(" ")[-1]
        params["camera_sn"] = device.get_info(rs.camera_info.serial_number)
        params_pool.append(params)

    pool = Pool(processes=len(devices))
    pool.map(record, params_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cameras", type=int, default=4, help="number of cameras")
    parser.add_argument("--timeout", type=int, default=2, help="how long to operate")
    parser.add_argument("--timeout_rest", type=int, default=10, help="how long to operate")
    parser.add_argument("--timeout_repeats", type=int, default=2, help="how long to operate")
    parser.add_argument("--width", type=int, default=640, help="how long to operate")
    parser.add_argument("--height", type=int, default=480, help="how long to operate")
    parser.add_argument("--fps", type=int, default=15, help="how long to operate")
    parser.add_argument("--post_proc", type=bool, default=True, help="how long to operate")

    parser.add_argument("--warmup_frames_d415", type=int, default=6, help="how long to operate")
    parser.add_argument("--warmup_frames_d435", type=int, default=10, help="how long to operate")
    parser.add_argument("--min_valid_frames", type=int, default=10, help="how long to operate")
    parser.add_argument("--depth_units_um", type=int, default=1000, help="how long to operate")
    parser.add_argument("--disparity_shift", type=int, default=0, help="how long to operate")
    parser.add_argument("--depth_clamp_min", type=int, default=300, help="how long to operate")
    parser.add_argument("--depth_clamp_max", type=int, default=2000, help="how long to operate")

    args = parser.parse_args()

    main(args)
