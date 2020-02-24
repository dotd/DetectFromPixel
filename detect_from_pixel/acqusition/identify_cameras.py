import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt


def catch_a_frame(camera_sn, width=640, height=480, fps=15):
    cfg = rs.config()
    cfg.enable_device(camera_sn)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    #cfg.enable_stream(rs.stream.depth, width, height, rs.format.rgb8, fps)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    aligner = rs.align(rs.stream.color)
    camera_profile = pipeline.get_active_profile()
    #device = camera_profile.get_device()

    frames = pipeline.wait_for_frames()
    rgb = frames.get_color_frame()
    rgb = np.asanyarray(rgb.get_data())
    return rgb


def identify_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()

    frames = list()
    for idx, device in enumerate(devices):
        sn = device.get_info(rs.camera_info.serial_number)
        print("idx={} name={} sn={}".format(idx, rs.camera_info.name, sn))
        rgb = catch_a_frame(sn)
        frames.append((sn, rgb))
        plt.figure()
        plt.imshow(rgb)
        plt.title("sn={}".format(sn))
        plt.show(block=False)
        plt.pause(0.01)

    plt.show(block=True)
    plt.pause(0.01)


if __name__ == "__main__":
    identify_cameras()