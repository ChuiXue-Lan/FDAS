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

import PIL
import cv2
import numpy as np
import torch
from PIL.Image import Image
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
    yolo2main_tts_dialog = Signal(int)  # 火源警告
    yolo2main_tts = Signal(int)

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

        self.fire_flag = False  # 是否识别到火焰
        self.isFirstTTS = True  # 是否为第一次识别到火焰

        # 运行时候的参数放这里
        self.start_time = 0  # 拿来算FPS的计数变量
        self.count = 0
        self.sum_of_count = 0
        self.class_num = 0
        self.total_frames = 0
        self.lock_id = 0

        # 设置线条样式    厚度 & 缩放大小
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )

    # 主要检测方法
    @smart_inference_mode()
    def run(self):
        # try:
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

            # 预热模型
            if not self.done_warmup:
                print('预热模型！')
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None

            # 开始检测
            print('开始检测！')
            count = 0  # 运行定位帧
            start_time = time.time()  # 用于计算帧率
            batch = iter(self.dataset)

            while True:
                # 终止检测
                if self.stop_dtc:
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
                    self.res_address(img_res, result, model)

                # 检测完成
                # if count + 1 >= all_count:
                #     if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                #         self.vid_writer[-1].release()  # 释放最终视频写入器
                #     self.yolo2main_status_msg.emit('Detection completed')
                #     break

        # except Exception as e:
        #     pass
        #     print(e)
        #     self.yolo2main_status_msg.emit('%s' % e)

    # 进行识别——并返回所有结果
    def res_address(self, img_res, result, model):  # self, img_res, result, height, width, model, out
        # 复制一份
        img_box = np.copy(img_res)  # 右边的图（会绘制标签！） img_res是原图-不会受影响
        img_trail = np.copy(img_res)  # 左边的图

        # 如果没有识别的：
        if result.boxes.id is None:
            # 目标都是0
            self.sum_of_count = 0
            self.class_num = 0
            labels_write = "暂未识别到目标！"
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

        # 写入txt——存储labels里的信息
        # if self.save_txt:
        #     with open(f'{self.save_txt_path}/result.txt', 'a') as f:
        #         f.write('当前时刻屏幕信息:' +
        #                 str(labels_write) +
        #                 f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
        #                 f' 路段通过的目标总数: {self.sum_of_count}')
        #         f.write('\n')

        # 预测视频写入本地 111
        # if self.save_res:
        #     out.write(img_box)

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
                self.yolo2main_tts_dialog.emit(1)
                self.yolo2main_tts.emit(1)

        # 进度条
        if '0' in self.source or 'rtsp' in self.source:
            self.yolo2main_progress.emit(0)
        else:
            self.progress_value = int(self.count / self.total_frames * 1000)
            self.yolo2main_progress.emit(self.progress_value)
        # 计算FPS
        self.count += 1
        if self.count % 3 == 0 and self.count >= 3:  # 计算FPS
            self.yolo2main_fps.emit(str(int(3 / (time.time() - self.start_time))))
            self.start_time = time.time()

    # def Interface_Refresh(self, im0, im0s, class_nums, target_nums):
    #     # Send test results 【发送信号 给 label 显示图像】
    #     self.yolo2main_res_img.emit(im0)  # after detection  ----------结果
    #     self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])  # Before testing
    #     # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
    #     self.yolo2main_class_num.emit(class_nums)
    #     self.yolo2main_target_num.emit(target_nums)

    def warmup_models(self):
        # 热身模型
        if not self.done_warmup:
            # 调用模型的 warmup 函数，其中 imgsz 参数为输入图像的大小
            # 如果模型使用 PyTorch，imgsz 参数应为 [batch_size, channels, height, width]
            # 如果模型使用 Triton，imgsz 参数应为 [height, width, channels, batch_size]
            self.model.warmup(
                imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            # 将 done_warmup 标记为 True，以标记模型已经热身过
            self.done_warmup = True
            print('热身完毕')

    # 获取 Annotator 实例
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    # 图像预处理
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # # uint8 转为 fp16/32
        img /= 255  # 0 - 255 转为 0.0 - 1.0
        return img

    # 预测结果后处理
    def postprocess(self, preds, img, orig_img):
        ### important
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    # 写入检测结果
    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # 为批次维度扩展
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f'{log_string}(no detections), '  # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n}~{self.model.names[int(c)]},"  # {'s' * (n > 1)}, "   # don't add 's'
        # now log_string is the classes 👆

        # 写
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.save_txt:  # 写入文件
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.save_res or self.args.save_crop or self.args.show or True:  # Add bbox to image(must)
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

        # 释放摄像头
    def release_capture(self):
        LoadStreams.capture = 'release'  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码
    # 不确定是不是这个包，报错进行更换

    def get_class_number(self):
        # 只识别火焰
        return 1
