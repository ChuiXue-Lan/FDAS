import threading
import pyttsx3
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
import classes.yolo
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
import json
import sys
import cv2
import os
from ui.utils.UIFunctions import *
from ui.utils.rtsp_win import Window
from ui.utils.tts_win import TTSWindow
# class YoloPredictor(BasePredictor, QObject):
#     # 信号定义
#     yolo2main_pre_img = Signal(np.ndarray)  # 原始图像信号
#     yolo2main_res_img = Signal(np.ndarray)  # 测试结果信号
#     yolo2main_status_msg = Signal(str)  # 检测/暂停/停止/测试完成/错误报告信号
#     yolo2main_fps = Signal(str)  # 帧率信号
#     yolo2main_labels = Signal(dict)  # 检测目标结果（每个类别的数量）
#     yolo2main_progress = Signal(int)  # 进度信号
#     yolo2main_class_num = Signal(int)  # 检测到的类别数
#     yolo2main_target_num = Signal(int)  # 检测到的目标数
#     yolo2main_tts_dialog = Signal(int)  # 火源警告
#     yolo2main_tts = Signal(int)
#
#     def __init__(self, cfg=DEFAULT_CFG, overrides=None):
#         super(YoloPredictor, self).__init__()
#         QObject.__init__(self)
#
#         # 获取配置
#         self.args = get_cfg(cfg, overrides)
#         project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
#         name = f'{self.args.mode}'
#         self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
#         self.done_warmup = False
#         if self.args.show:
#             self.args.show = check_imshow(warn=True)
#
#         # GUI 参数
#         self.used_model_name = None  # 使用的检测模型名称
#         self.new_model_name = None  # 实时更改的模型
#         self.source = ''  # 输入源
#         self.stop_dtc = False  # 终止检测
#         self.continue_dtc = True  # 暂停检测
#         self.save_res = False  # 保存测试结果
#         self.save_txt = False  # 保存标签(txt)文件
#         self.iou_thres = 0.45  # IoU 阈值
#         self.conf_thres = 0.25  # 置信度阈值
#         self.speed_thres = 10  # 延迟，毫秒
#         self.labels_dict = {}  # 返回结果的字典
#         self.progress_value = 0  # 进度条
#
#         # 仅在配置设置完成后可用
#         self.model = None
#         self.data = self.args.data  # data_dict
#         self.imgsz = None
#         self.device = None
#         self.dataset = None
#         self.vid_path, self.vid_writer = None, None
#         self.annotator = None
#         self.data_path = None
#         self.source_type = None
#         self.batch = None
#         self.callbacks = defaultdict(list, callbacks.default_callbacks)  # 添加回调函数
#         callbacks.add_integration_callbacks(self)
#
#         self.fire_flag = False  # 是否识别到火焰
#         self.isFirstTTS = True  # 是否为第一次识别到火焰
#
#     # 主要检测方法
#     @smart_inference_mode()
#     def run(self):
#         # try:
#         if self.args.verbose:
#             LOGGER.info('')
#
#         # 设置模型
#         self.yolo2main_status_msg.emit('加载模型...')
#         if not self.model:
#             self.setup_model(self.new_model_name)
#             self.used_model_name = self.new_model_name
#
#         # 设置输入源
#         self.setup_source(self.source if self.source is not None else self.args.source)
#
#         # 检查保存路径/标签
#         if self.save_res or self.save_txt:
#             (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
#
#         # 预热模型
#         if not self.done_warmup:
#             self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
#             self.done_warmup = True
#
#         self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
#
#         # 开始检测
#         count = 0  # 运行定位帧
#         start_time = time.time()  # 用于计算帧率
#         batch = iter(self.dataset)
#
#         while True:
#             # 终止检测
#             if self.stop_dtc:
#                 if isinstance(self.vid_writer[-1], cv2.VideoWriter):
#                     self.vid_writer[-1].release()  # 释放最终视频写入器
#                 self.yolo2main_status_msg.emit('Detection terminated!')
#                 break
#
#             # # 在检测过程中更改模型
#             # if self.used_model_name != self.new_model_name:
#             #     # self.yolo2main_status_msg.emit('Change Model...')
#             #     self.setup_model(self.new_model_name)
#             #     self.used_model_name = self.new_model_name
#
#             # 暂停开关
#             if self.continue_dtc:
#                 # time.sleep(0.001)
#                 self.yolo2main_status_msg.emit('Detecting...')
#                 batch = next(self.dataset)  # 下一个数据
#
#                 self.batch = batch
#                 path, im, im0s, vid_cap, s = batch
#
#                 visualize = increment_path(self.save_dir / Path(path).stem,
#                                            mkdir=True) if self.args.visualize else False
#
#                 # 计算完成度和帧率（待优化）
#                 count += 1  # 帧计数 +1
#                 if vid_cap:
#                     all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
#                 else:
#                     all_count = 1
#                 self.progress_value = int(count / all_count * 1000)  # 进度条(0~1000)
#                 if count % 5 == 0 and count >= 5:  # 每5帧计算一次帧率
#                     self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
#                     start_time = time.time()
#
#                 # 预处理
#                 with self.dt[0]:
#                     im = self.preprocess(im)
#                     if len(im.shape) == 3:
#                         im = im[None]  # 为批次维度扩展
#                 # 推理
#                 with self.dt[1]:
#                     preds = self.model(im, augment=self.args.augment, visualize=visualize)
#                 # 后处理
#                 with self.dt[2]:
#                     self.results = self.postprocess(preds, im, im0s)
#
#                 # 可视化、保存、写入结果
#                 n = len(im)  # 待改进: 支持多张图像
#                 for i in range(n):
#                     self.results[i].speed = {
#                         'preprocess': self.dt[0].dt * 1E3 / n,
#                         'inference': self.dt[1].dt * 1E3 / n,
#                         'postprocess': self.dt[2].dt * 1E3 / n}
#                     p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
#                         else (path, im0s.copy())
#                     p = Path(p)  # 源目录
#
#                     # s:::   video 1/1 (6/6557) 'path':
#                     # must, to get boxs\labels
#                     label_str = self.write_results(i, self.results, (p, im, im0))  # 标签字符串   /// original :s +=
#
#                     # 标签和数量
#                     class_nums = 0
#                     target_nums = 0
#                     self.labels_dict = {}
#                     if 'no detections' in label_str:
#                         pass
#                     else:
#                         for ii in label_str.split(',')[:-1]:
#                             nums, label_name = ii.split('~')
#                             self.labels_dict[label_name] = int(nums)
#                             target_nums += int(nums)
#                             # print('警告！检测到火源！！')
#                             self.fire_flag = True
#                             if self.fire_flag & self.isFirstTTS:
#                                 self.isFirstTTS = False
#                                 self.yolo2main_tts_dialog.emit(1)
#                                 self.yolo2main_tts.emit(1)
#
#                             class_nums += 1
#
#                     # 保存图像或视频结果
#                     if self.save_res:
#                         self.save_preds(vid_cap, i, str(self.save_dir / p.name))
#
#                     # 发送测试结果信号
#                     self.yolo2main_res_img.emit(im0)  # 检测后
#                     self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])  # 检测前
#                     # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
#                     self.yolo2main_class_num.emit(class_nums)
#                     self.yolo2main_target_num.emit(target_nums)
#
#                     if self.speed_thres != 0:
#                         time.sleep(self.speed_thres / 1000)  # 延迟，毫秒
#
#                 self.yolo2main_progress.emit(self.progress_value)  # 进度条
#
#             # 检测完成
#             if count + 1 >= all_count:
#                 if isinstance(self.vid_writer[-1], cv2.VideoWriter):
#                     self.vid_writer[-1].release()  # 释放最终视频写入器
#                 self.yolo2main_status_msg.emit('Detection completed')
#                 break
#
#     # except Exception as e:
#     #     pass
#     #     print(e)
#     #     self.yolo2main_status_msg.emit('%s' % e)
#
#     @smart_inference_mode()
#     def camera_run(self):
#         try:
#             if self.args.verbose:
#                 LOGGER.info('')
#
#             # set model
#             self.yolo2main_status_msg.emit('加载模型...')
#
#             if not self.model:
#                 self.setup_model(self.new_model_name)
#                 self.used_model_name = self.new_model_name
#
#             # 检查保存路径/标签
#             if self.save_res or self.save_txt:
#                 (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
#
#             self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
#             # 创建名为 dt 的实例变量，用于存储一个元组，并将其初始化为包含三个对象 ops.Profile() 的元组。
#             # ops.Profile() 是指从 ops 模块中导入名为 Profile() 的对象。
#
#             # start detection
#
#             count = 0  # run location frame
#             start_time = time.time()  # used to calculate the frame rate
#
#             # ------------------
#             self.capture = cv2.VideoCapture(0)  # 创建一个 OpenCV 视频捕获对象
#
#             while True:
#                 time.sleep(0.06)  # 休眠 60 毫秒（0.06 秒）
#                 ret, frame = self.capture.read()  # 捕获摄像头的图像
#                 if ret:  # 如果读取成功
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     self.capture_frame = Image.fromarray(frame_rgb)
#
#                 print(self.capture_frame)
#                 self.setup_source(self.capture_frame)
#
#                 # warmup model
#                 # 热身模型
#                 if not self.done_warmup:
#                     # 调用模型的 warmup 函数，其中 imgsz 参数为输入图像的大小
#                     # 如果模型使用 PyTorch，imgsz 参数应为 [batch_size, channels, height, width]
#                     # 如果模型使用 Triton，imgsz 参数应为 [height, width, channels, batch_size]
#                     self.model.warmup(
#                         imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
#                     # 将 done_warmup 标记为 True，以标记模型已经热身过
#                     self.done_warmup = True
#
#                 batch = iter(self.dataset)
#
#                 # pause switch  用于控制程序的暂停和继续
#                 if self.continue_dtc:
#                     # time.sleep(0.001)
#                     self.yolo2main_status_msg.emit('Detecting...')
#                     batch = next(self.dataset)  # next data
#
#                     self.batch = batch
#                     path, im, im0s, vid_cap, s = batch
#                     visualize = increment_path(self.save_dir / Path(path).stem,
#                                                mkdir=True) if self.args.visualize else False
#
#                     # Calculation completion and frame rate (to be optimized)
#                     count += 1  # frame count +1
#                     all_count = 1000  # all_count 可以调整！！！
#                     self.progress_value = int(count / all_count * 1000)  # progress bar(0~1000)
#                     if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
#                         self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
#                         start_time = time.time()
#
#                     # preprocess
#                     # self.dt 包含了三个 DetectorTime 类型的对象，表示预处理、推理和后处理所花费的时间
#
#                     ## 使用 with 语句记录下下一行代码所花费的时间，self.dt[0] 表示记录预处理操作所花费的时间。
#                     with self.dt[0]:
#                         # 调用 self.preprocess 方法对图像进行处理，并将处理后的图像赋值给 im 变量。
#                         im = self.preprocess(im)
#                         # 如果 im 的维度为 3（RGB 图像），则表示这是一张单张图像，需要将其扩展成 4 维，加上 batch 维度。
#                         if len(im.shape) == 3:
#                             im = im[None]  # expand for batch dim  扩大批量调暗
#                     # inference
#                     with self.dt[1]:
#                         # 调用模型对图像进行推理，并将结果赋值给 preds 变量。
#                         preds = self.model(im, augment=self.args.augment, visualize=visualize)
#                     # postprocess
#                     with self.dt[2]:
#                         # 调用 self.postprocess 方法对推理结果进行后处理，并将结果保存到 self.results 变量中。
#                         # 其中 preds 是模型的预测结果，im 是模型输入的图像，而 im0s 是原始图像的大小。
#                         self.results = self.postprocess(preds, im, im0s)
#
#                     # visualize, save, write results
#                     n = len(im)  # To be improved: support multiple img
#
#                     for i in range(n):
#                         self.results[i].speed = {
#                             'preprocess': self.dt[0].dt * 1E3 / n,
#                             'inference': self.dt[1].dt * 1E3 / n,
#                             'postprocess': self.dt[2].dt * 1E3 / n
#                         }
#                         p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
#                             else (path, im0s.copy())
#
#                         p = Path(p)  # the source dir
#
#                         # s:::   video 1/1 (6/6557) 'path':
#                         # must, to get boxs\labels
#                         label_str = self.write_results(i, self.results, (p, im, im0))  # labels   /// original :s +=
#
#                         # labels and nums dict
#                         class_nums = 0
#                         target_nums = 0
#                         self.labels_dict = {}
#                         if 'no detections' in label_str:
#                             pass
#                         else:
#                             for ii in label_str.split(',')[:-1]:
#                                 nums, label_name = ii.split('~')
#
#                                 li = nums.split(':')[-1]
#
#                                 print(li)
#
#                                 self.labels_dict[label_name] = int(li)
#                                 target_nums += int(li)
#                                 class_nums += 1
#
#                         # save img or video result
#                         if self.save_res:
#                             self.save_preds(vid_cap, i, str(self.save_dir / p.name))
#
#                         # Send test results 【发送信号 给 label 显示图像】
#                         self.yolo2main_res_img.emit(im0)  # after detection  ----------结果
#                         self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])  # Before testing
#                         # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
#                         self.yolo2main_class_num.emit(class_nums)
#                         self.yolo2main_target_num.emit(target_nums)
#
#                         if self.speed_thres != 0:
#                             time.sleep(self.speed_thres / 1000)  # delay , ms
#
#                     self.yolo2main_progress.emit(self.progress_value)  # progress bar
#
#                 # Detection completed
#                 if count + 1 >= all_count:
#                     if isinstance(self.vid_writer[-1], cv2.VideoWriter):
#                         self.vid_writer[-1].release()  # release final video writer
#                     self.yolo2main_status_msg.emit('Detection completed')
#                     break
#
#         except Exception as e:
#
#             self.capture.release()  # 释放视频设备
#             print('error:', e)
#             self.yolo2main_status_msg.emit('%s' % e)
#
#     # 获取 Annotator 实例
#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
#
#     # 图像预处理
#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()  # # uint8 转为 fp16/32
#         img /= 255  # 0 - 255 转为 0.0 - 1.0
#         return img
#
#     # 预测结果后处理
#     def postprocess(self, preds, img, orig_img):
#         ### important
#         preds = ops.non_max_suppression(preds,
#                                         self.conf_thres,
#                                         self.iou_thres,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det,
#                                         classes=self.args.classes)
#
#         results = []
#         for i, pred in enumerate(preds):
#             orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
#             shape = orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
#             path, _, _, _, _ = self.batch
#             img_path = path[i] if isinstance(path, list) else path
#             results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
#         return results
#
#     # 写入检测结果
#     def write_results(self, idx, results, batch):
#         p, im, im0 = batch
#         log_string = ''
#         if len(im.shape) == 3:
#             im = im[None]  # 为批次维度扩展
#         self.seen += 1
#         imc = im0.copy() if self.args.save_crop else im0
#         if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
#         self.annotator = self.get_annotator(im0)
#
#         det = results[idx].boxes  # TODO: make boxes inherit from tensors
#
#         if len(det) == 0:
#             return f'{log_string}(no detections), '  # if no, send this~~
#
#         for c in det.cls.unique():
#             n = (det.cls == c).sum()  # detections per class
#             log_string += f"{n}~{self.model.names[int(c)]},"  # {'s' * (n > 1)}, "   # don't add 's'
#         # now log_string is the classes 👆
#
#         # 写
#         for d in reversed(det):
#             cls, conf = d.cls.squeeze(), d.conf.squeeze()
#             if self.save_txt:  # 写入文件
#                 line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
#                     if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
#             if self.save_res or self.args.save_crop or self.args.show or True:  # Add bbox to image(must)
#                 c = int(cls)  # integer class
#                 name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
#                 label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
#                 self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
#             if self.args.save_crop:
#                 save_one_box(d.xyxy,
#                              imc,
#                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                              BGR=True)
#
#         return log_string

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # 主窗口发出的信号，用于触发Yolo实例执行

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 基本界面设置
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 圆角透明
        self.setWindowFlags(Qt.FramelessWindowHint)  # 设置窗口标志: 隐藏窗口边框
        UIFuncitons.uiDefinitions(self)
        # 显示模块阴影
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # 读取模型文件夹
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))  # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)  # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # 语音警示
        self.engine = pyttsx3.init()  # 语音警示
        self.engine_switch = True  # 循环警告

        # Yolo-v8 线程
        self.yolo_predict = classes.yolo.YoloPredictor()  # 创建一个Yolo实例
        self.select_model = self.model_box.currentText()  # 默认模型
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.yolo_thread = QThread()  # 创建Yolo线程
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)                            
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.yolo_predict.yolo2main_tts_dialog.connect(self.voice_announcement)
        self.yolo_predict.yolo2main_tts.connect(self.tts)
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # 模型参数
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # 提示窗口初始化
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # 选择检测源
        self.src_file_button.clicked.connect(self.open_src_file)  # 选择本地文件
        self.src_cam_button.clicked.connect(self.chose_cam)  # chose_cam
        self.src_rtsp_button.clicked.connect(self.chose_rtsp)  # chose_rtsp

        # 开始测试按钮
        self.run_button.clicked.connect(self.run_or_continue)  # 开始/暂停测试按钮
        self.stop_button.clicked.connect(self.stop)  # 终止测试按钮

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # 保存图像选项
        self.save_txt_button.toggled.connect(self.is_save_txt)  # 保存标签选项
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # 左侧导航按钮
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # 右上角设置按钮

        # 初始化
        self.load_config()

    # 主窗口显示原始图像和检测结果的静态方法
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持原始数据比例
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 控制开始/暂停
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('请在开始检测前选择视频源...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # 启动按钮
                self.save_txt_button.setEnabled(False)  # 在开始检测后禁止勾选并保存
                self.save_res_button.setEnabled(False)
                self.show_status('检测中...')
                self.yolo_predict.continue_dtc = True  # 控制Yolo是否暂停
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("暂停...")
                self.run_button.setChecked(False)  # 启动按钮

    # 底部状态栏信息
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 结束进程
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 结束进程
            self.pre_video.clear()  # 清空图像显示
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # 选择本地文件
    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']

        # 如果配置文件中的路径不存在，则使用当前工作目录
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    # 选择摄像头源 have one bug
    def chose_cam(self):
        self.yolo_predict.camera_run()
        # try:
        #     self.stop()
        #     MessageBox(
        #         self.close_button, title='注意', text='正在加载摄像头...', time=2000, auto=True).exec()
        #     # 获取本地摄像头数量
        #     _, cams = Camera().get_cam_num()
        #     popMenu = QMenu()
        #     popMenu.setFixedWidth(self.src_cam_button.width())
        #     popMenu.setStyleSheet('''
        #                                     QMenu {
        #                                     font-size: 16px;
        #                                     font-family: "Microsoft YaHei UI";
        #                                     font-weight: light;
        #                                     color:white;
        #                                     padding-left: 5px;
        #                                     padding-right: 5px;
        #                                     padding-top: 4px;
        #                                     padding-bottom: 4px;
        #                                     border-style: solid;
        #                                     border-width: 0px;
        #                                     border-color: rgba(255, 255, 255, 255);
        #                                     border-radius: 3px;
        #                                     background-color: rgba(200, 200, 200,50);}
        #                                     ''')
        #
        #     for cam in cams:
        #         exec("action_%s = QAction('%s')" % (cam, cam))
        #         exec("popMenu.addAction(action_%s)" % cam)
        #
        #     x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
        #     y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
        #     y = y + self.src_cam_button.frameGeometry().height()
        #     pos = QPoint(x, y)
        #     action = popMenu.exec(pos)
        #     if action:
        #         self.yolo_predict.source = action.text()
        #         self.show_status('加载摄像头：{}'.format(action.text()))
        #
        # # except Exception as e:
        # #     self.show_status('%s' % e)

    # 选择网络源
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://192.168.2.72:8554/h264_ulaw.sdp"
            # ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def voice_announcement(self):  # TTS:TextToSpeech
        self.TTS_window = TTSWindow()
        self.TTS_window.show()
        tts_dialog = threading.Thread(target=self.tts_dialog_show)
        tts_dialog.start()
        self.TTS_window.checkButton.clicked.connect(lambda: self.check_fire(self.TTS_window))
        self.TTS_window.cancelButton.clicked.connect(lambda: self.close_tts_dialog(self.TTS_window))

    def tts_dialog_show(self):
        self.TTS_window.exec()

    def tts(self):
        # 语音警告线程
        tts_thread = threading.Thread(target=self.speak)
        tts_thread.start()

    def speak(self):
        self.engine_switch = True  # 开启播报
        while self.engine_switch:
            self.engine.say('警告！发现火源！')
            self.engine.runAndWait()

    def check_fire(self, TTS_window):
        self.engine_switch = False
        # 恢复现场
        self.yolo_predict.fire_flag = False  # 是否识别到火焰
        self.yolo_predict.isFirstTTS = True  # 是否为第一次识别到火焰
        TTS_window.close()

    def close_tts_dialog(self, TTS_window):
        self.engine_switch = False
        # 恢复现场
        self.yolo_predict.fire_flag = False  # 是否识别到火焰
        self.yolo_predict.isFirstTTS = True  # 是否为第一次识别到火焰
        TTS_window.close()

    # 加载网络源
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('加载 rtsp：{}'.format(ip))
            self.rtsp_window.close()
            # 展示

        except Exception as e:
            self.show_status('%s' % e)

    # 保存测试结果按钮--图片/视频
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('注意：运行图像结果不会被保存。')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('注意：运行图像结果将被保存。')
            self.yolo_predict.save_res = True

    # 保存测试结果按钮 -- 标签 (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('注意：标签结果不会被保存。')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('注意：标签结果将被保存。')
            self.yolo_predict.save_txt = True

    # 配置初始化  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # 终止按钮和相关状态
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # 结束线程
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # 恢复启动键
        self.save_res_button.setEnabled(True)  # 可以使用保存按钮
        self.save_txt_button.setEnabled(True)  # 可以使用保存按钮
        self.pre_video.clear()  # 清空图像显示
        self.res_video.clear()  # 清空图像显示
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # 改变检测参数
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))  # 值变化，改变滑块
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)  # 滑块值变化，改变值
            self.show_status('IOU 阈值: %s' % str(x / 100))
            self.yolo_predict.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.show_status('Conf 阈值: %s' % str(x / 100))
            self.yolo_predict.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('延迟：%s 毫秒' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # 更改模型
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('更改模型：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # 循环监视模型文件更改
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # 必须在比较之前进行排序，否则列表会一直刷新
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # 获取鼠标位置（用于按住标题栏拖动窗口）
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # 调整窗口大小底部和右边缘拖动时的优化调整
    def resizeEvent(self, event):
        # 更新大小手柄
        UIFuncitons.resize_grips(self)

    # 关闭事件 退出线程，保存设置
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState() == Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState() == Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # 在关闭之前退出进程
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='注意', text='正在退出，请稍候...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
