#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/12  16:52
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 含mp4附件的邮件发送
# -*- encoding:utf-8 -*-
# =============================================================================
# 导入必须的Python3模块或者包
# =============================================================================
import ssl
import smtplib
from datetime import datetime
# 导入构造邮件头部标题的方法
from email.header import Header
# 导入解析发件人地址和收件人地址的方法
from email.utils import parseaddr
# 导入构造发件人地址和收件人地址的方法
from email.utils import formataddr
# 导入构造文本格式的邮件的方法，MIMEText对象代表文本邮件对象
from email.mime.text import MIMEText
# 导入构造音视频的方法，MIMEAudio对象代表音视频邮件对象
from email.mime.audio import MIMEAudio
# 导入构造图片的方法，MIMEImage对象代表图片邮件对象
from email.mime.image import MIMEImage
# 导入构造MIME格式邮件的方法，MIME格式邮件即富文本邮件
from email.mime.multipart import MIMEMultipart

# =============================================================================
# 设置QQ邮箱的登录信息以及发件人、收件人、抄送人、密送人等
# =============================================================================
# QQ邮件服务器的安全端口465，注意是数字格式
smtp_port = 465
# QQ邮件服务器的地址
smtp_host = 'smtp.qq.com'
# 登录QQ邮件服务器的用户名
user_name = '2358369326@qq.com'
# 此处填写自己申请到的登录QQ邮件服务器的授权码
user_pass = 'lrvlwkondmjhdihd'
# 发件人邮箱昵称和邮箱地址
sender = '火灾警报<2358369326@qq.com>'
# 发件人邮箱地址
sender_addr = '2358369326@qq.com'
# 收件人邮箱昵称和邮箱地址，列表中可以包含多个，给email模块使用
receivers = ['菠萝吹雪<2358369326@qq.com>']
# 收件人邮箱地址，列表中可以包含多个，给smtplib模块使用
rece_addrs = ['2358369326@qq.com']
# 抄送人邮箱昵称和邮箱地址，列表中可以包含多个，给email模块使用
ccers = ['菠萝吹雪<2358369326@qq.com>']
# 抄送人邮箱地址，列表中可以包含多个，给smtplib模块使用
cc_addrs = ['2358369326@qq.com']
# 密送人邮箱昵称和邮箱地址，列表中可以包含多个，给email模块使用
bccers = ['菠萝吹雪<2358369326@qq.com>']
# 密送人邮箱地址，列表中可以包含多个，给smtplib模块使用
bcc_addrs = ['2358369326@qq.com']
# =============================================================================
# 构造html格式的邮件内容
# =============================================================================
html_msg = """
<!DOCTYPE html>
<html lang="zh-cmn-Hans">
    <head>
        <meta charset="UTF-8">
        <title>火灾警报</title>
        <meta name="description" content="警告！发现火情！">
    </head>
    <body>
        <p>警告！发现火情！请尽快处理！</p>
        <p>报警时间：{}</p>
        <br>
        <br>
        <video width="300" height="300" controls>
            <source src="cid:fire_video" type="vedio/mp4" />
            对不起，您的浏览器不支持video标签
        </video>
    </body>
</html>
"""
# 用字符串格式化函数format生成邮件主题字符串
email_sub = '邮件主题是:{email_sub}'.format(email_sub='火灾警报')


# =============================================================================
# 自定义一个构造QQ邮件的类
# =============================================================================
class CreateEmail(object):
    # =========================================================================
    # 类实例的初始化函数，实例化时获取必须的参数
    # =========================================================================
    def __init__(self, email_sub, html_msg, sender, receivers, ccers, bccers):
        """
        1.self.email_sub是邮件主题，一个字符串类型的数据
        2.self.sender是发件人邮箱昵称和邮箱地址，一个字符串类型的数据
        3.self.receivers是邮件收件人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        4.self.ccers是邮件抄送人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        5.self.bccers是邮件密送人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        """
        self.email_sub = email_sub
        self.html_msg = html_msg
        self.sender = sender
        self.receivers = receivers
        self.ccers = ccers
        self.bccers = bccers
        # 实例化一个邮件整体
        self.email_msg = MIMEMultipart('mixed')

    # =========================================================================
    # format_addr是一个类函数，用于解析收件人、抄送人、密送人的邮件地址并重构
    # =========================================================================
    @staticmethod
    def format_addr(name_addr):
        """
        1.name_addr是一个字符串类型的数据，它是原始的邮箱昵称和邮箱地址
        原始的邮箱昵称和邮箱地址不能在QQ邮箱中正常显示，影响邮件美观度
        2.format_addr函数返回的是重构后的可以在QQ中正常显示的邮箱昵称和邮箱地址
        """
        name, addr = parseaddr(name_addr)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    # =========================================================================
    # send_date是一个类函数，用于构造邮件的发送日期
    # =========================================================================
    @staticmethod
    def send_date():
        """返回当前的日期和时间"""
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return date_time

    # =========================================================================
    # 构造邮件头的函数，可以一次性加入多个收件人、抄送人、密送人
    # =========================================================================
    def email_header(self):
        """构造邮件头的函数，没有返回值，函数体内已将邮件头格式化完毕"""
        # 构造邮件头中主题，突出邮件内容重点
        self.email_msg['Subject'] = Header(self.email_sub, 'utf-8').encode()
        # 构造邮件头中时间，显示发送邮件的时间，一般不需要构造，QQ邮箱自动生成
        # self.email_msg['Date'] = Header(self.send_date(), 'utf-8').encode()
        # 构造邮件头中的发件人，包括昵称和邮箱账号
        self.email_msg['From'] = self.format_addr(self.sender)
        # =====================================================================
        # 构造邮件头中的收件人，包括昵称和邮箱账号
        # =====================================================================
        # 定义一个列表，存放格式化之后的收件人地址
        format_rec_addrs = []
        # 采用for循环一次性将多个收件人地址格式化
        for receiver_addr in self.receivers:
            format_rec_addrs.append(self.format_addr(receiver_addr))
        self.email_msg['To'] = ','.join(format_rec_addrs)
        # =====================================================================
        # 构造邮件头中的抄送人，包括昵称和邮箱账号
        # =====================================================================
        # 定义一个列表，存放格式化之后的抄送人地址
        format_cc_addrs = []
        # 采用for循环一次性将多个抄送人地址格式化
        for cc_addr in self.ccers:
            format_cc_addrs.append(self.format_addr(cc_addr))
        self.email_msg['Cc'] = ','.join(format_cc_addrs)
        # =====================================================================
        # 构造邮件头中的密送人，包括昵称和邮箱账号
        # =====================================================================
        # 定义一个列表，存放格式化之后的密送人地址
        format_bcc_addrs = []
        # 采用for循环一次性将多个密送人地址格式化
        for bcc_addr in self.bccers:
            format_bcc_addrs.append(self.format_addr(bcc_addr))
        self.email_msg['Bcc'] = ','.join(format_bcc_addrs)

    # =========================================================================
    # 构造邮件体的函数，可以添加超链接、图片、音频、视频
    # =========================================================================
    def email_body(self):
        """构造邮件体的函数，没有返回值，函数体内已将邮件体构造完毕"""
        # 添加发现时间到正文
        self.html_msg = self.html_msg.format(self.send_date())
        # 采用'utf-8'编码并将html格式的内容附加到mixed邮件体
        self.email_msg.attach(MIMEText(self.html_msg, 'html', 'utf-8'))
        # =====================================================================
        # 向html正文中插入视频的方法如下，视频路径中最好不要有中文
        # =====================================================================
        # 读取一个指定文件夹下的视频文件，注意文件大小不能超过QQ邮箱邮件大小限制，否则发送失败
        with open(r'D:\Pycharm\FDAS-main\runs\results\fire_video.mp4',
                  'rb') as video_file:
            # 注意指定视频文件的子格式_subtype，防止自动识别失败无法正常加载
            video_msg = MIMEAudio(video_file.read(), _subtype='mp4')
        # 定义视频ID，以使html文件正确引用视频
        video_msg.add_header('Content-ID', 'fire_video')
        # 将video视频附加到邮件整体
        self.email_msg.attach(video_msg)

    # =========================================================================
    # 构造邮件附件的函数，可以添加图片、文本、Excel文件、Word文件、PDF文件、音频、视频附件
    # =========================================================================
    def email_att(self):
        """构造邮件附件的函数，没有返回值，函数体内已将邮件附件构造完毕"""
        # # =====================================================================
        # # 向邮件中添加文本文件附件的方法如下，注意路径格式，原文件名可为中文或英文
        # # =====================================================================
        # # 以二进制方式打开指定文件夹中的文本文件
        # with open(r'C:\Users\Admin\Pictures\Saved Pictures\安泽频道.txt',
        #           'rb') as text_file:
        #     # 用打开的文件构造一个文本对象，注意是MIMEText对象，而非MIMEImage对象，
        #     # 数据流为base64格式
        #     text_att = MIMEText(text_file.read(), 'base64', 'utf-8')
        # # 设置文本的内容格式
        # text_att['Content-Type'] = 'application/octet-stream'
        # # 设置文本附件的名称，可以与原文件名不相同，因为windows系统中文件名是用gbk编码，
        # # 所以附件名称为中文时的写法如下
        # text_att.add_header('Content-Disposition', 'attachment',
        #                     filename=('gbk', '', 'xxx.txt'))
        # # 当附件名称非中文时的写法如下
        # # text_att['Content-Disposition'] = 'attachment; filename='xxxx.txt''
        # # 向MIMEMultipart对象中添加文本附件对象
        # self.email_msg.attach(text_att)
        # =====================================================================
        # 向邮件中添加视频附件的方法如下，注意路径格式，原文件名可为中文或英文
        # =====================================================================
        # 以二进制方式打开指定文件夹中的视频文件
        with open(r'D:\Pycharm\FDAS-main\runs\results\fire_video.mp4',  # TODO：改为相对路径
                  'rb') as video_file:
            # 用打开的文件构造一个视频文件对象，注意是MIMEText对象，而非MIMEImage对象，
            # 数据流为base64格式
            video_att = MIMEText(video_file.read(), 'base64', 'utf-8')
        # 设置视频文件的内容格式
        video_att['Content-Type'] = 'application/octet-stream'
        # 设置视频文件附件的名称，可以与原文件名不相同，因为windows系统中文件名是用gbk编码，
        # 所以附件名称为中文时的写法如下
        video_att.add_header('Content-Disposition', 'attachment',
                             filename=('gbk', '', '火情视频.mp4'))
        # 当附件名称非中文时的写法如下
        # video_att['Content-Disposition'] = 'attachment; filename='xxx.mp4''
        # 向MIMEMultipart对象中添加视频文件附件对象
        self.email_msg.attach(video_att)

    def return_email(self):
        """返回字符串化后的邮件，供smtplib模块中的sendmail函数调用"""
        self.email_header()
        self.email_body()
        self.email_att()
        # email_msg.as_string()是将MIMEText对象或MIMEMultipart对象变为str
        return self.email_msg.as_string()


# =============================================================================
# 自定义一个登录并发送QQ邮件的类
# =============================================================================
class SendEmail(object):
    # =========================================================================
    # 类实例的初始化函数，实例化时获取必须的参数
    # =========================================================================
    def __init__(self, smtp_port, smtp_host, user_name, user_pass,
                 sender_addr, rece_addrs, cc_addrs, bcc_addrs, email_msg):
        """
        1.self.smtp_port是邮箱端口，一个整数类型的数据
        2.self.smtp_host是QQ邮箱服务器地址，一个字符串类型的数据
        3.self.user_name是发件人邮箱登录账户，一个字符串列表类型的数据
        4.self.user_pass是发件人邮箱登录授权码，一个字符串列表类型的数据
        5.self.sender_addr是发件人邮箱地址，一个字符串类型的数据
        6.self.rece_addrs是收件人邮箱地址，一个字符串列表类型的数据
        7.self.cc_addrs是抄送人邮箱地址，一个字符串列表类型的数据
        8.self.bcc_addrs是密送人邮箱地址，一个字符串列表类型的数据
        9.self.email_msg是字符串化的邮件整体，一个字符串类型的数据
        """
        self.smtp_port = smtp_port
        self.smtp_host = smtp_host
        self.user_name = user_name
        self.user_pass = user_pass
        self.sender_addr = sender_addr
        self.rece_addrs = rece_addrs
        self.cc_addrs = cc_addrs
        self.bcc_addrs = bcc_addrs
        self.email_msg = email_msg
        # 设置安全上下文
        self.context = ssl.create_default_context()

    # =========================================================================
    # 登录并发送邮件的函数
    # =========================================================================
    def login_send(self):
        # 采用with结构登录邮箱并发送邮件，执行结束后可自动断开与邮件服务器的连接
        with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port,
                              context=self.context) as email_svr:
            # 输入QQ邮箱的账号和授权码后登录
            email_svr.login(self.user_name, self.user_pass)
            # 邮箱登录成功后即可发送邮件
            # 将收件人邮箱地址、抄送人邮箱地址、密送人邮箱地址以加号连接，
            # 注意地址中不能包括邮箱昵称
            email_svr.sendmail(self.sender_addr, self.rece_addrs +
                               self.cc_addrs + self.bcc_addrs, self.email_msg)

def email_send():
    # 实例化一个创建email邮件的类
    create_email = CreateEmail(email_sub, html_msg, sender, receivers,
                               ccers, bccers)
    # 获得字符串化后的邮件对象
    email_msg = create_email.return_email()
    # 实例化一个登录并发送QQ邮件的类
    send_email = SendEmail(smtp_port, smtp_host, user_name, user_pass,
                           sender_addr, rece_addrs, cc_addrs, bcc_addrs,
                           email_msg)
    # =========================================================================
    # 采用try结构发送邮件
    # =========================================================================
    try:
        # 发送QQ邮件
        send_email.login_send()
    # 如果发生可预知的smtp类错误，则执行下面代码
    except smtplib.SMTPException as e:
        print('smtp发生错误，邮件发送失败，错误信息为：', e)
    # 如果发生不可知的异常则执行下面语句结构中的代码
    except Exception as e:
        print('发生不可知的错误，错误信息为：', e)
    # 如果没发生异常则执行else语句结构中的代码
    else:
        print('邮件发送未发生任何异常，一切正常！')
    # 无论是否发生异常，均执行finally语句结构中的代码
    finally:
        print('邮件发送程序已执行完毕！')

if __name__ == '__main__':
    # email_send = SendEmail()
    # email_send.email_send()
    email_send()
