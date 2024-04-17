#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/17  18:36
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
import threading
import time
from collections import defaultdict
from pathlib import Path

# import PIL
# from PIL.Image import Image
import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadStreams
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ui.utils.UIFunctions import *
import datetime
import supervision as sv

video_id = 0


class YoloPredictor(BasePredictor, QObject):
    # 信号定义
    yolo2main_pre_img = Signal(np.ndarray)  # 原始图像信号
    yolo2main_res_img = Signal(np.ndarray)  # 测试结果信号
    yolo2main_status_msg = Signal(str)  # 检测/暂停/停止/测试完成/错误报告信号
    yolo2main_fps = Signal(str)  # 帧率信号
    yolo2main_labels = Signal(dict)  # 检测目标结果（每个类别的数量）
    yolo2main_progress = Signal(int)  # 进度信号
    yolo2main_class_num = Signal(int)  # 检测到的类别数
    yolo2main_target_num = Signal(int)  # 检测到的目标数
    yolo2main_tts_dialog = Signal()  # 火源警告对话框
    yolo2main_tts = Signal()  # 火源警告
    yolo2main_email_send = Signal()  # 邮件发送

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        # 获取配置
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI 参数
        self.used_model_name = None  # 使用的检测模型名称
        self.new_model_name = None  # 实时更改的模型
        self.source = ''  # 输入源
        self.stop_dtc = False  # 终止检测
        self.continue_dtc = True  # 暂停检测
        self.save_res = False  # 保存测试结果
        self.save_txt = False  # 保存标签(txt)文件
        self.save_path = "runs/results"
        self.iou_thres = 0.45  # IoU 阈值
        self.conf_thres = 0.25  # 置信度阈值
        self.speed_thres = 10  # 延迟，毫秒
        self.labels_dict = {}  # 返回结果的字典
        self.progress_value = 0  # 进度条

        # 仅在配置设置完成后可用
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # 添加回调函数
        callbacks.add_integration_callbacks(self)

        # 运行时候的参数放这里
        self.fire_flag = False  # 是否识别到火焰
        self.isFirstTTS = True  # 是否为第一次识别到火焰
        self.start_time = 0  # 拿来算FPS的计数变量
        self.count = 0
        self.sum_of_count = 0
        self.class_num = 0
        self.total_frames = 0
        self.lock_id = 0
        self.fire_video = None  # 截取视频
        self.fire_video_height = 0
        self.fire_video_width = 0

        # 设置线条样式    厚度 & 缩放大小
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )

    # 主要检测方法
    @smart_inference_mode()
    def run(self):
        global video_id
        try:
            if self.args.verbose:
                LOGGER.info('')

            # 设置模型
            self.yolo2main_status_msg.emit('加载模型...')
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # 设置输入源
            print('输入源：' + str(self.source) + '\n')
            self.setup_source(self.source if self.source is not None else self.args.source)
            model = YOLO(self.new_model_name)

            # 使用OpenCV读取视频——获取进度条???
            if 'mp4' in self.source or 'avi' in self.source or 'mkv' in self.source or 'flv' in self.source or 'mov' in self.source:
                cap = cv2.VideoCapture(self.source)
                self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

            # 检查保存路径/标签
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 获取数据源 （不同的类型获取不同的数据源）
            iter_model = iter(
                model.track(source=self.source, show=False, stream=True, iou=self.iou_thres, conf=self.conf_thres))

            # 如果保存，则创建写入对象
            img_res, result, height, width = self.recognize_res(iter_model)
            self.fire_video_height = height
            self.fire_video_width = width
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # TODO：VideoWriter_fourcc找不到？
            out = None  # 视频写出变量
            if self.save_res:
                out = cv2.VideoWriter(f'{self.save_path}/video_result_{video_id}.mp4', fourcc, 25,
                                      (width, height), True)  # 保存检测视频的路径

            # 预热模型
            if not self.done_warmup:
                print('预热模型！')
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None

            # 开始检测
            print('开始检测！')
            while True:
                try:
                    # 终止检测
                    if self.stop_dtc:
                        # 开启视频保存，则保存视频并将id自增
                        if self.save_res:
                            if out:
                                out.release()
                                video_id += 1
                        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                            self.vid_writer[-1].release()  # 释放最终视频写入器
                        self.yolo2main_status_msg.emit('检测终止!')
                        self.release_capture()
                        break

                    # # 在检测过程中更改模型
                    # if self.used_model_name != self.new_model_name:
                    #     # self.yolo2main_status_msg.emit('Change Model...')
                    #     self.setup_model(self.new_model_name)
                    #     self.used_model_name = self.new_model_name

                    # 暂停开关
                    if self.continue_dtc:
                        img_res, result, height, width = self.recognize_res(iter_model)
                        self.res_address(img_res, result, out)

                except StopIteration:  # 重点，没有这个结束时会报错
                    if self.save_res:
                        out.release()
                        video_id += 1
                    self.yolo2main_status_msg.emit('检测完成')
                    self.yolo2main_progress.emit(1000)
                    cv2.destroyAllWindows()
                    self.source = None
                    break

        except Exception as e:
            pass
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)

    # 进行识别——并返回所有结果
    def res_address(self, img_res, result, out):  # self, img_res, result, height, width, model, out
        # 复制一份
        img_box = np.copy(img_res)  # 右边的图（会绘制标签！） img_res是原图-不会受影响
        img_trail = np.copy(img_res)  # 左边的图
        self.fire_video = img_box

        # 如果没有识别的：
        if result.boxes.id is None:
            # 目标都是0
            self.sum_of_count = 0
            self.class_num = 0
        # 如果有识别的
        else:
            detections = sv.Detections.from_yolov8(result)
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            # id 、位置、目标总数
            self.labels_dict = {}
            self.class_num = self.get_class_number()  # 类别数
            id = detections.tracker_id  # id
            xyxy = detections.xyxy  # 位置
            self.sum_of_count = len(id)  # 目标总数

            # 画标签到图像上
            if self.class_num != 0:
                img_box = self.box_annotator.annotate(scene=img_box, detections=detections)

            # labels_write, img_box = self.creat_labels(detections, img_box, model)

            # 相关数据信息
            labels_write = [
                f"目标ID: {tracker_id} 目标类别: {class_id} 置信度: {confidence:0.2f}"
                for _, _, confidence, class_id, tracker_id in detections
            ]

            # 写入txt——存储labels里的信息 # TODO: 之后改为存储到数据库
            if self.save_txt:
                with open(f'{self.save_path}/result.txt', 'a') as f:
                    f.write(f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
                            f' 识别到的目标总数: {self.sum_of_count}' +
                            ' 当前时刻屏幕信息:' + str(labels_write))
                    f.write('\n')

        # 预测视频写入本地
        if self.save_res:
            out.write(img_box)

        # 传递信号给主窗口
        self.emit_res(img_trail, img_box)

    # 识别结果处理
    def recognize_res(self, iter_model):
        # 检测 ---然后获取有用的数据
        result = next(iter_model)  # 这里是检测的核心，每次循环都会检测一帧图像,可以自行打印result看看里面有哪些key可以用
        img_res = result.orig_img  # 原图
        height, width, _ = img_res.shape

        return img_res, result, height, width

    # 信号发送区
    def emit_res(self, img_trail, img_box):
        time.sleep(self.speed_thres / 1000)  # 缓冲
        # 轨迹图像（左边）
        self.yolo2main_pre_img.emit(img_trail)
        # 标签图（右边）
        self.yolo2main_res_img.emit(img_box)
        # 总类别数量 、 总目标数
        self.yolo2main_class_num.emit(self.class_num)
        self.yolo2main_target_num.emit(self.sum_of_count)

        # 报警
        if self.sum_of_count > 0:
            self.fire_flag = True
            if self.fire_flag & self.isFirstTTS:
                self.isFirstTTS = False
                self.yolo2main_tts_dialog.emit()
                self.yolo2main_tts.emit()
                self.yolo2main_email_send.emit()

        # 进度条
        if '0' in self.source or 'rtsp' in self.source:
            self.yolo2main_progress.emit(1)
        else:
            self.progress_value = int(self.count / self.total_frames * 1000)
            self.yolo2main_progress.emit(self.progress_value)
        # 计算FPS
        self.count += 1
        if self.count % 3 == 0 and self.count >= 3:  # 计算FPS
            self.yolo2main_fps.emit(str(int(3 / (time.time() - self.start_time))))
            self.start_time = time.time()

    # 获取 Annotator 实例
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    # 释放摄像头
    def release_capture(self):
        LoadStreams.capture = 'release'  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码

    # 不确定是不是这个包，报错进行更换

    def get_class_number(self):
        # 只识别火焰，则返回1即可
        return 1

    # 截取视频
    def get_mp4(self):
        flag = True

        # 计时
        def keep_time(flag):
            time.sleep(8)  # 截取八秒视频
            flag = False

        print("yolo: get_mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'{self.save_path}/fire_video.mp4', fourcc, 25,
                              (self.fire_video_width, self.fire_video_height), True)  # 保存检测视频的路径

        keep_time_thread = threading.Thread(target=keep_time, args=(flag,))
        keep_time_thread.start()

        count = 0
        while flag:
            out.write(self.fire_video)
            count += 1  # 保证循环一定会停止
            if count >= 250:
                break

        out.release()
        print("finished get_mp4")


