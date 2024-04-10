#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/29  20:55
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
import threading
import pyttsx3
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
import classes.yolo
from classes.camera import Camera
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
import json
import sys
import cv2
import os
from ui.utils.UIFunctions import *
from ui.utils.rtsp_win import Window
from ui.utils.tts_win import TTSWindow


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

        # 摄像头选择
        self.camIndex = 0  # 默认为0号（本机）摄像头
        self.isCam = False

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
        self.src_cam_button.clicked.connect(self.camera_select)  # chose_cam
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
            # 使图像在GUI上显示出来
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 控制开始/暂停
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('请在开始检测前选择视频源...')
            self.run_button.setChecked(False)
        else:
            print('视频源加载成功...')
            self.yolo_predict.stop_dtc = False  # 线程开始

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

    # camera选择
    def camera_select(self):
        # try:
        # 关闭YOLO线程
        self.stop()
        # 获取本地摄像头数量
        _, cams = Camera().get_cam_num()
        popMenu = QMenu()
        popMenu.setFixedWidth(self.src_cam_button.width())
        popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(16,155,226,50);
                                            }
                                            ''')

        for cam in cams:
            exec("action_%s = QAction('%s 号摄像头')" % (cam, cam))
            exec("popMenu.addAction(action_%s)" % cam)
        pos = QCursor.pos()
        action = popMenu.exec(pos)

        # 设置摄像头来源
        if action:
            str_temp = ''
            selected_stream_source = str_temp.join(filter(str.isdigit, action.text()))  # 获取摄像头号，去除非数字字符
            self.yolo_predict.source = selected_stream_source
            self.show_status(f'摄像头设备:{action.text()}')
            self.isCam = True
            # DialogOver(parent=self, text=f"当前摄像头为: {action.text()}", title="摄像头选择成功", flags="success")

    # 选择网络源
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = 'rtsp://admin:123456@192.168.0.106:8554/live'
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

        # 关闭摄像头
        if self.isCam:
            # 遍历可能的摄像头索引号
            for i in range(10):  # 假设摄像头索引号在 0 到 9 之间
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"摄像头 {i} 已经打开。")
                    cap.release()  # 释放资源
            self.isCam = False

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
