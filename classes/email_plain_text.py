#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/14  11:06
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 纯文本邮件发送
# -*- encoding:utf-8 -*-
import smtplib
import ssl
from datetime import datetime
# 导入构造邮件头部标题的方法
from email.header import Header
# 导入构造MIME格式邮件的方法，MIME格式邮件即富文本邮件
from email.mime.multipart import MIMEMultipart
# 导入构造文本格式的邮件的方法，MIMEText对象代表文本邮件对象
from email.mime.text import MIMEText
# 导入构造发件人地址和收件人地址的方法
from email.utils import formataddr
# 导入解析发件人地址和收件人地址的方法
from email.utils import parseaddr


class EmailPlainText(object):
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
    # 构造MIMEMultipart('alternative')类型的邮件体，用于包含两段text文件
    # =============================================================================
    email_msg = MIMEMultipart('alternative')
    # 获取当前的日期和时间
    date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 用字符串格式化函数format生成邮件主题字符串
    email_sub = '邮件主题是:{email_sub},发送时间是:{send_time}'.format(
        email_sub='测试邮件', send_time=date_time)

    # =============================================================================
    # 解析收件人、抄送人、密送人的邮件地址并重构
    # =============================================================================
    def format_addr(self, name_addr):
        """
        1.name_addr是一个字符串类型的数据，它是原始的邮箱昵称和邮箱地址
        原始的邮箱昵称和邮箱地址不能在QQ邮箱中正常显示，影响邮件美观度
        2.format_addr函数返回的是重构后的可以在QQ中正常显示的邮箱昵称和邮箱地址
        """
        name, addr = parseaddr(name_addr)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    # =============================================================================
    # 构造邮件头的函数，可以一次性加入多个收件人、抄送人、密送人
    # =============================================================================
    def email_header(self):
        """
        1.email_sub是邮件主题，一个字符串类型的数据
        2.sender是发件人邮箱昵称和邮箱地址，一个字符串类型的数据
        3.receivers是邮件收件人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        4.ccers是邮件抄送人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        5.bccers是邮件密送人邮箱昵称和邮箱地址，一个字符串列表类型的数据
        6.函数没有返回值，函数体内已将邮件头格式化完毕
        """
        # 构造邮件头中主题，突出邮件内容重点
        self.email_msg['Subject'] = Header(self.email_sub, 'utf-8').encode()
        # 构造邮件头中的发件人，包括昵称和邮箱账号
        self.email_msg['From'] = self.format_addr(self.sender)
        # =========================================================================
        # 构造邮件头中的收件人，包括昵称和邮箱账号
        # =========================================================================
        # 定义一个列表，存放格式化之后的收件人地址
        format_rec_addrs = []
        # 采用for循环一次性将多个收件人地址格式化
        for receiver_addr in self.receivers:
            format_rec_addrs.append(self.format_addr(receiver_addr))
        self.email_msg['To'] = ','.join(format_rec_addrs)
        # =========================================================================
        # 构造邮件头中的抄送人，包括昵称和邮箱账号
        # =========================================================================
        # 定义一个列表，存放格式化之后的抄送人地址
        format_cc_addrs = []
        # 采用for循环一次性将多个抄送人地址格式化
        for cc_addr in self.ccers:
            format_cc_addrs.append(self.format_addr(cc_addr))
        self.email_msg['Cc'] = ','.join(format_cc_addrs)
        # =========================================================================
        # 构造邮件头中的密送人，包括昵称和邮箱账号
        # =========================================================================
        # 定义一个列表，存放格式化之后的密送人地址
        format_bcc_addrs = []
        # 采用for循环一次性将多个密送人地址格式化
        for bcc_addr in self.bccers:
            format_bcc_addrs.append(self.format_addr(bcc_addr))
        self.email_msg['Bcc'] = ','.join(format_bcc_addrs)

    def email_send(self):
        # 执行构造邮件头的函数
        self.email_header()
        # =============================================================================
        # 构造邮件，邮件体内容是两段文本，采用'utf-8'
        # =============================================================================
        # 第一段文本的内容
        text_msg01 = "警告！火灾检测系统发现火情！请即使处理！\n"
        # 构造第一段纯文本格式的邮件内容，采用'utf-8'编码并附加到alternative邮件体
        self.email_msg.attach(MIMEText(text_msg01, 'plain', 'utf-8'))
        # 第二段文本的内容
        text_msg02 = "发现时间：" + self.date_time
        # 构造第二段纯文本格式的邮件内容，采用'utf-8'编码并附加到alternative邮件体
        self.email_msg.attach(MIMEText(text_msg02, 'plain', 'utf-8'))
        # =============================================================================
        # 设置安全上下文
        # =============================================================================
        context = ssl.create_default_context()
        # =============================================================================
        # 采用try结构发送邮件
        # =============================================================================
        try:
            # 采用with结构登录邮箱并发送邮件，执行结束后可自动断开与邮件服务器的连接
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as email_svr:
                # 输入QQ邮箱的账号和授权码后登录
                email_svr.login(self.user_name, self.user_pass)
                # 邮箱登录成功后即可发送邮件
                # 将收件人邮箱地址、抄送人邮箱地址、密送人邮箱地址以加号连接，注意地址中不能包括邮箱昵称
                # email_msg.as_string()是将MIMEText对象或MIMEMultipart对象变为str
                email_svr.sendmail(self.sender_addr, self.rece_addrs + self.cc_addrs + self.bcc_addrs,
                                   self.email_msg.as_string())
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
    email_plain_text = EmailPlainText()
    email_plain_text.email_send()
