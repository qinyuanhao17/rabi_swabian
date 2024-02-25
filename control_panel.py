import sys
import os
import time
import gc
from PyQt5 import QtGui
from pympler import asizeof
import pythoncom
import pyvisa
import rabi_swabian_ui
import pandas as pd
import numpy as np
import pyqtgraph as pg
from threading import Thread, active_count
from ctypes import *
#import JSON-RPC Pulse Streamer wrapper class, to use Google-RPC import from pulsestreamer.grpc
from pulsestreamer import PulseStreamer, Sequence, OutputState, findPulseStreamers
# impofr timetagger 
os.environ['TIMETAGGER_INSTALL_PATH'] = 'C:\Program Files\Swabian Instruments\Time Tagger'
import TimeTagger as tt

from PyQt5.QtGui import QIcon, QPixmap, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsDropShadowEffect, QFileDialog, QDesktopWidget, QVBoxLayout

class ConfigureChannels():
    def __init__(self):
        super().__init__()
        self._pulser_channels = {
            'ch_aom': 0, # output channel 0
            'ch_switch': 1, # output channel 1
            'ch_daq': 2, # output channel 2
            'ch_sync': 3 # output channel 3
        }
        self._timetagger_channels = {
            'click_channel': 1,
            'start_channel':2,
            'next_channel':-2,
            'sync_channel':tt.CHANNEL_UNUSED,
        }    
    @property
    def pulser_channels(self):
        return self._pulser_channels
    @property
    def timetagger_channels(self):
        return self._timetagger_channels
    
class MyWindow(rabi_swabian_ui.Ui_Form, QWidget):

    rf_info_msg = pyqtSignal(str)
    pulse_streamer_info_msg = pyqtSignal(str)
    data_processing_info_msg = pyqtSignal(str)
    tcspc_data_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)


    def __init__(self):

        super().__init__()

        # init UI
        self.setupUi(self)
        self.ui_width = int(QDesktopWidget().availableGeometry().size().width()*0.75)
        self.ui_height = int(QDesktopWidget().availableGeometry().size().height()*0.8)
        self.resize(self.ui_width, self.ui_height)
        center_pointer = QDesktopWidget().availableGeometry().center()
        x = center_pointer.x()
        y = center_pointer.y()
        old_x, old_y, width, height = self.frameGeometry().getRect()
        self.move(int(x - width / 2), int(y - height / 2))

        # set flag off and widget translucent
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # set window blur
        self.render_shadow()
        
        # init window button signal
        self.window_btn_signal()

        '''
        RF init
        '''
        # Init RF combobox ui
        self.rf_cbx_test()
        
        # Init RF setup info ui
        self.rf_info_ui()

        # Init RF signal
        self.my_rf_signal()
        '''
        Confugure channels
        '''
        channel_config = ConfigureChannels()
        pulser_channels = channel_config.pulser_channels
        timetagger_channels = channel_config.timetagger_channels
        self._channels = {**pulser_channels, **timetagger_channels}

        # print(self._channels)
        '''
        PULSER init
        '''
        
        self.pulse_streamer_singal_init()
        self.pulse_streamer_info_ui()
        self.pulse_streamer_info_msg.connect(self.pulse_streamer_slot)
        self.pulsestreamer_on_activate()
        '''
        TimeTagger init
        '''
        self.timetagger_on_activate()
        '''
        Data processing init
        '''
        self.plot_ui_init()
        self.data_processing_signal()
        self.data_processing_info_ui()
        
    def data_processing_signal(self):

        # Message signal
        self.data_processing_info_msg.connect(self.data_processing_slot)
        self.tcspc_data_signal.connect(self.plot_result)
        # Scroll area updating signal
        self.data_processing_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.data_processing_scroll.verticalScrollBar().setValue(
                self.data_processing_scroll.verticalScrollBar().maximum()
            )
        )
        # plot signal
        self.repeat_cycle_spbx.valueChanged.connect(self.process_plot_data)
        self.repeat_cycle_spbx.valueChanged.connect(self.rabi_cycling)
        self.save_plot_data_btn.clicked.connect(self.save_plot_data)
        # self.hist_num_cbx.currentTextChanged.connect(self.process_plot_data)

        # infinite line signal
        self.signal_start_spbx.editingFinished.connect(self.reset_infinite_line_pos)
        self.signal_span_spbx.editingFinished.connect(self.reset_infinite_line_pos)
        self.ref_start_spbx.editingFinished.connect(self.reset_infinite_line_pos)
        self.ref_span_spbx.editingFinished.connect(self.reset_infinite_line_pos)

        self.data_start.sigPositionChangeFinished.connect(self.reset_infinite_line_spbx_value)
        self.data_stop.sigPositionChangeFinished.connect(self.reset_infinite_line_spbx_value)
        self.ref_start.sigPositionChangeFinished.connect(self.reset_infinite_line_spbx_value)
        self.ref_stop.sigPositionChangeFinished.connect(self.reset_infinite_line_spbx_value)
    def save_plot_data(self):
        
        pass
    
    def process_plot_data(self):
        if hasattr(self, '_tcspc_data_container') and int(self.repeat_cycle_spbx.value()):
            
            dataType = self.hist_num_cbx.currentText()
            rabi_dataType = self.rabi_data_type_cbx.currentText()
            thread = Thread(
                target= self.process_plot_data_thread,
                args=(dataType, rabi_dataType),
            )
            thread.start()
    
    def rabi_plot_data_thread(self, dataType):
        pass

    def generate_rabi_data(self, data, dataType):
        signal_start_pos = round(self.data_start.value())
        signal_stop_pos = round(self.data_stop.value())
        ref_start_pos = round(self.ref_start.value())
        ref_stop_pos = round(self.ref_stop.value())
        repeat_count = int(self.repeat_cycle_spbx.value())
        rabi_dict = {
            'sum': lambda x: np.sum(x[:,signal_start_pos:signal_stop_pos],axis=1),
            'mean': lambda x: np.sum(x[:,signal_start_pos:signal_stop_pos],axis=1)/repeat_count,
            'mean_norm': lambda x: np.sum(x[:,signal_start_pos:signal_stop_pos],axis=1)/np.sum(x[:,ref_start_pos:ref_stop_pos],axis=1),
            'reference': lambda x: np.sum(x[:,signal_start_pos:signal_stop_pos],axis=1)-np.sum(x[:,ref_start_pos:ref_stop_pos],axis=1)
        }
        return rabi_dict[dataType](data)
    def process_plot_data_thread(self, dataType, rabi_dataType):
        tcspc_data = self._tcspc_data_container[1:]
        start, stop, step, numpoints = self.start_stop_step()
        rabi_index = np.arange(start,stop+step,step)
        assert len(rabi_index)==numpoints,'Rabi index number error!'
        rabi_intensity = self.generate_rabi_data(data=tcspc_data, dataType=rabi_dataType)
        
        if dataType == 'SUM':
            self.tcspc_data_signal.emit(self._tcspc_index/1000, np.sum(tcspc_data, axis=0), rabi_index, rabi_intensity)

        else:
            self.tcspc_data_signal.emit(self._tcspc_index/1000, tcspc_data[int(dataType)-1], rabi_index, rabi_intensity)
        
    def plot_result(self, tcspc_x, tcspc_y, rabi_index, rabi_intensity):

        '''Plot tcspc data'''    
        start_time = time.time() 
        self.tcspc_curve.setData(tcspc_x, tcspc_y)
        self.rabi_curve.setData(rabi_index, rabi_intensity)

        end_time = time.time()
        print(f'plot time: {end_time-start_time}') 
                     
    def data_processing_info_ui(self):

        self.data_processing_msg.setWordWrap(True)  # 自动换行
        self.data_processing_msg.setAlignment(Qt.AlignTop)  # 靠上

        # # 用于存放消息
        self.data_processing_msg_history = []

    def data_processing_slot(self, msg):

        # print(msg)
        self.data_processing_msg_history.append(msg)
        self.data_processing_msg.setText("<br>".join(self.data_processing_msg_history))
        self.data_processing_msg.resize(700, self.data_processing_msg.frameSize().height() + 20)
        self.data_processing_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容
    
    def generate_infinite_line(self, pos=0, pen=None, label=None):

        line = pg.InfiniteLine(
                pos=pos, 
                angle=90, 
                pen=pen, 
                movable=True, 
                bounds=(0,None), 
                hoverPen=None, 
                label=label, 
                labelOpts={'position': 0.99}, 
                span=(0, 1), 
                markers=None, 
                name=None
            )
        return line
    def create_plot_widget(self, xlabel, ylabel, title, frame, infiniteLine=False):
        plot = pg.PlotWidget(enableAutoRange=True, useOpenGL=True)
        graph_widget_layout = QVBoxLayout()
        graph_widget_layout.addWidget(plot)
        frame.setLayout(graph_widget_layout)
        plot.setLabel("left", ylabel)
        plot.setLabel("bottom", xlabel)
        plot.setTitle(title, color='k')
        plot.setBackground(background=None)
        plot.getAxis('left').setPen('k')
        plot.getAxis('left').setTextPen('k')
        plot.getAxis('bottom').setPen('k')
        plot.getAxis('bottom').setTextPen('k')
        plot.getAxis('top').setPen('k')
        plot.getAxis('right').setPen('k')
        plot.showAxes(True)
        plot.showGrid(x=False, y=True)
        curve = plot.plot(pen=pg.mkPen(color=(255,85,48), width=2))        
        if infiniteLine == True:
            signal_start_pos = int(self.signal_start_spbx.value())
            signal_span = int(self.signal_span_spbx.value())
            ref_start_pos = int(self.ref_start_spbx.value())
            ref_span = int(self.ref_span_spbx.value())
            signal_stop_pos = signal_start_pos + signal_span
            ref_stop_pos = ref_start_pos + ref_span

            data_pen = pg.mkPen(color='b', width=1)
            ref_pen = pg.mkPen(color='g', width=1)

            self.data_start = self.generate_infinite_line(pen=data_pen,pos=signal_start_pos,label='start')
            self.data_stop = self.generate_infinite_line(pen=data_pen,pos=signal_stop_pos,label='stop')
            self.ref_start = self.generate_infinite_line(pen=ref_pen,pos=ref_start_pos,label='start')
            self.ref_stop = self.generate_infinite_line(pen=ref_pen,pos=ref_stop_pos,label='stop')
            plot.addItem(self.data_start)
            plot.addItem(self.data_stop)
            plot.addItem(self.ref_start)
            plot.addItem(self.ref_stop)

        return curve
    def reset_infinite_line_spbx_value(self):
        signal_start_pos = round(self.data_start.value())
        signal_stop_pos = round(self.data_stop.value())
        ref_start_pos = round(self.ref_start.value())
        ref_stop_pos = round(self.ref_stop.value())

        signal_span = signal_stop_pos - signal_start_pos
        ref_span = ref_stop_pos - ref_start_pos
        
        if signal_span > 0 and ref_span > 0:
            self.signal_start_spbx.setValue(signal_start_pos)
            self.signal_span_spbx.setValue(signal_span)
            self.ref_start_spbx.setValue(ref_start_pos)
            self.ref_span_spbx.setValue(ref_span)
    def reset_infinite_line_pos(self):
        signal_start_pos = int(self.signal_start_spbx.value())
        signal_span = int(self.signal_span_spbx.value())
        ref_start_pos = int(self.ref_start_spbx.value())
        ref_span = int(self.ref_span_spbx.value())
        signal_stop_pos = signal_start_pos + signal_span
        ref_stop_pos = ref_start_pos + ref_span
        self.data_start.setValue(signal_start_pos)
        self.data_stop.setValue(signal_stop_pos)
        self.ref_start.setValue(ref_start_pos)
        self.ref_stop.setValue(ref_stop_pos)
    def plot_ui_init(self):
        self.tcspc_curve = self.create_plot_widget(
            xlabel='Time bins (ns)',
            ylabel='Counts',
            title='TCSPC Data',
            frame=self.tcspc_graph_frame,
            infiniteLine=True
        )
        self.rabi_curve = self.create_plot_widget(
            xlabel='Time (ns)',
            ylabel="Contrast (a.u.)",
            title='Rabi',
            frame=self.rabi_graph_frame,
        )

    def timetagger_on_activate(self):
        self._tagger = tt.createTimeTagger()
        self._tagger.reset()

        # set trigger level to 1.5V
        self._tagger.setTriggerLevel(1, 1.5)
        self._tagger.sync()
    def timetagger_on_deactivate(self):
        self.pulsed.stop()
        self.pulsed.clear()
        self.pulsed = None
        tt.freeTimeTagger(self._tagger)
    def pulse_streamer_singal_init(self):

        # ASG scroll area scrollbar signal
        self.pulse_streamer_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.pulse_streamer_scroll.verticalScrollBar().setValue(
                self.pulse_streamer_scroll.verticalScrollBar().maximum()
            )
        )

        self.rabi_set_btn.clicked.connect(lambda: self.set_pulse_and_count(**self._channels))
        self.rabi_start_btn.clicked.connect(self.rabi_start)
        self.rabi_stop_btn.clicked.connect(self.rabi_stop)
    def rabi_stop(self):

        self.pulsed.stop()
        self.pulsed.clear()
        self.pulser.reset()
 
        self._stopConstant = True
        gc.collect()
        
    def rabi_start(self):
        self.repeat_cycle_spbx.setValue(0)
        print('start clicked')
        '''
        Reset pulser and tagger
        '''
        self.pulsed.stop()
        self.pulsed.clear()
        self.pulser.reset()
        '''
        Init a tcspc data container
        '''
        self._tcspc_data_container = np.array([])
        self._tcspc_index = np.array([])
        '''
        Start Rabi
        '''
        self._stopConstant = False
        
        self.pulsed.start()
        self.pulsed.setMaxCounts(self._int_cycles)
        time.sleep(0.5)
        final = OutputState([self._channels['ch_aom']],0,0)
        self.pulser.stream(self.seq, self._int_cycles+1, final)

        thread = Thread(
            target=self.count_data_thread_func
        )
        thread.start()
    def rabi_cycling(self):

        if self._stopConstant == False and int(self.repeat_cycle_spbx.value()):        
            self.pulsed.start()
            self.pulsed.setMaxCounts(self._int_cycles)
            time.sleep(0.2)
            final = OutputState([self._channels['ch_aom']],0,0)
            self.pulser.stream(self.seq, self._int_cycles+1, final)

            thread = Thread(
                target=self.count_data_thread_func
            )
            thread.start()

    def count_data_thread_func(self):
                
        while True:
            if self.pulsed.ready():
                data = self.pulsed.getData()
                if self._tcspc_data_container.size == 0:
                    self._tcspc_data_container = data
                else:
                    self._tcspc_data_container += data
                del data
                gc.collect()
                self._tcspc_index = self.pulsed.getIndex()                
                break
            time.sleep(0.5)
        self.pulsed.stop()
        self.pulsed.clear()
        self.pulser.reset()
        repeat_cycle = int(self.repeat_cycle_spbx.value())
        self.repeat_cycle_spbx.setValue(repeat_cycle+1)
       
    def write_hist_cbx(self,mw_times):
        self.hist_num_cbx.clear()
        self.hist_num_cbx.addItems(['SUM']+[str(i) for i in range(1,len(mw_times))])
    def start_stop_step(self):
        start = int(self.rabi_start_spbx.value())
        stop = int(self.rabi_stop_spbx.value())
        step = int(self.rabi_step_spbx.value())
        num_points = int((stop - start)/step) + 1
        return start, stop, step, num_points
    def set_pulse_and_count(self, ch_aom, ch_switch, ch_daq, ch_sync, click_channel, start_channel, next_channel, sync_channel):

        start, stop, step, num_points = self.start_stop_step()

        laser_time = int(self.laser_time_spbx.value())*1000 # in ns
        laser_delay = int(self.laser_delay_spbx.value())
        wait_time = int(self.wait_time_spbx.value())
        mw_times = range(start,stop+2*step,step) # 第一个laser time是没有任何信息的
        self.write_hist_cbx(mw_times)
        if len(range(1,len(mw_times))) != num_points:
            self.pulse_streamer_info_msg.emit('MW list lens error!')
        #define digital levels
        HIGH=1
        LOW=0
        seq_aom=[]
        seq_switch=[]
        seq_daq=[]
        seq_gate=[]
        #define pulse patterns for each channels
        # simply add more pulses with ', (time, HIGH/LOW)'
        for mw_time in mw_times:
            seq_aom += [(laser_time, HIGH), (wait_time, LOW), (mw_time,LOW)]
            seq_switch += [(laser_time, LOW), (wait_time, LOW), (mw_time, HIGH)]
            seq_daq += [(laser_time, HIGH), (wait_time, LOW), (mw_time,LOW)]
            seq_gate += [(laser_time+laser_delay, HIGH), (wait_time-laser_delay, LOW), (mw_time,LOW)]
        
        #create the sequence
        self.seq = Sequence()
        
        #set digital channels
        self.seq.setDigital(ch_aom, seq_aom)
        self.seq.setDigital(ch_switch, seq_switch)
        self.seq.setDigital(ch_daq, seq_daq)
        self.seq.setDigital(ch_sync, seq_gate)

        self.seq.plot()

        ''''
        Configer Timetagger
        '''

        # self._number_of_gates = number_of_gates
        bin_width = int(self.bin_width_cbx.currentText()) * 1000 #in ps
        record_length = int(laser_time+laser_delay)*1000 # in ps
        n_bins = record_length/bin_width
        n_histograms = sum([tup[1] for tup in seq_aom]) # get the total number of high levels of aom
        n_mws = sum([tup[1] for tup in seq_switch])
        # print(n_histograms)
        # print(num_points)
        assert n_mws == n_histograms, 'High level number error!'
        assert record_length % bin_width == 0, 'Bins number error!'

        
        if sync_channel == tt.CHANNEL_UNUSED:
            self.pulsed = tt.TimeDifferences(
                tagger=self._tagger,
                click_channel=click_channel,
                start_channel=start_channel,
                next_channel=start_channel,
                sync_channel=sync_channel,
                binwidth=bin_width, # In ps
                n_bins=int(n_bins),
                n_histograms=n_histograms
            )
        else:
            self.pulse_streamer_info_msg.emit('sync_channel error!')
        self._int_cycles = int(self.int_cycle_spbx.value())
        # self.pulsed.setMaxCounts(self._int_cycles)
        self.pulsed.stop() 
        self.pulsed.clear()
        
        
    def timetagger_channels(self):
        channels = ConfigureChannels.timetagger_channels 
        return channels 
    
    def pulsestreamer_on_activate(self):
        devices = findPulseStreamers()
        # DHCP is activated in factory settings
        if devices !=[]:
            ip = devices[0][0]
        else:
            # if discovery failed try to connect by the default hostname
            # IP address of the pulse streamer (default hostname is 'pulsestreamer')
            self.pulse_streamer_info_msg.emit("No Pulse Streamer found")

        #connect to the pulse streamer
        self.pulser = PulseStreamer(ip)

        # Print serial number and FPGA-ID
        self.pulse_streamer_info_msg.emit('Serial: ' + self.pulser.getSerial())
        self.pulse_streamer_info_msg.emit('FPGA ID: ' + self.pulser.getFPGAID())

    def pulsestreamer_on_deactivate(self):
        self.pulser.reset()
    def pulse_streamer_info_ui(self):

        self.pulse_streamer_msg.setWordWrap(True)  # 自动换行
        self.pulse_streamer_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.pulse_streamer_msg_history = []

    def pulse_streamer_slot(self, msg):

        # print(msg)
        self.pulse_streamer_msg_history.append(msg)
        self.pulse_streamer_msg.setText("<br>".join(self.pulse_streamer_msg_history))
        self.pulse_streamer_msg.resize(700, self.pulse_streamer_msg.frameSize().height() + 20)
        self.pulse_streamer_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    '''Set window ui'''
    def window_btn_signal(self):
        # window button sigmal
        self.close_btn.clicked.connect(self.close)
        self.max_btn.clicked.connect(self.maxornorm)
        self.min_btn.clicked.connect(self.showMinimized)
        
    #create window blur
    def render_shadow(self):
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)  # 偏移
        self.shadow.setBlurRadius(30)  # 阴影半径
        self.shadow.setColor(QColor(128, 128, 255))  # 阴影颜色
        self.mainwidget.setGraphicsEffect(self.shadow)  # 将设置套用到widget窗口中

    def maxornorm(self):
        if self.isMaximized():
            self.showNormal()
            self.norm_icon = QIcon()
            self.norm_icon.addPixmap(QPixmap(":/my_icons/images/icons/max.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.norm_icon)
        else:
            self.showMaximized()
            self.max_icon = QIcon()
            self.max_icon.addPixmap(QPixmap(":/my_icons/images/icons/norm.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.max_icon)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = QPoint
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标
        
    def mouseMoveEvent(self, QMouseEvent):
        m_position = QPoint
        m_position = QMouseEvent.globalPos() - self.pos()
        width = QDesktopWidget().availableGeometry().size().width()
        height = QDesktopWidget().availableGeometry().size().height()
        if m_position.x() < width*0.7 and m_position.y() < height*0.06:
            self.m_flag = True
            if Qt.LeftButton and self.m_flag:                
                pos_x = int(self.m_Position.x())
                pos_y = int(self.m_Position.y())
                if pos_x < width*0.7 and pos_y < height*0.06:           
                    self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
                    QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    '''
    RF CONTROL
    '''
    def rf_info_ui(self):

        self.rf_msg.setWordWrap(True)  # 自动换行
        self.rf_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.rf_msg_history = []

    def rf_slot(self, msg):

        # print(msg)
        self.rf_msg_history.append(msg)
        self.rf_msg.setText("<br>".join(self.rf_msg_history))
        self.rf_msg.resize(700, self.rf_msg.frameSize().height() + 20)
        self.rf_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    def my_rf_signal(self):

        #open button signal
        self.rf_connect_btn.clicked.connect(self.boot_rf)

        #message signal
        self.rf_info_msg.connect(self.rf_slot)

        # RF scroll area scrollbar signal
        self.rf_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.rf_scroll.verticalScrollBar().setValue(
                self.rf_scroll.verticalScrollBar().maximum()
            )
        )

        # combobox restore signal
        self.rf_visa_rst_btn.clicked.connect(self.rf_cbx_test)

        # RF On button signal
        self.rf_ply_stp_btn.clicked.connect(self.rf_ply_stp)


    def rf_cbx_test(self):
        
        self.rf_cbx.clear()
        self.rm = pyvisa.ResourceManager()
        self.ls = self.rm.list_resources()
        self.rf_cbx.addItems(self.ls)

    def boot_rf(self):
        
        # Boot RF generator
        self.rf_port = self.rf_cbx.currentText()
        # print(self.rf_port)
        self._gpib_connection = self.rm.open_resource(self.rf_port)
        self._gpib_connection.write_termination = '\n'
        instrument_info = self._gpib_connection.query('*IDN?')
        
        # # 恢复出厂设置
        # self.fac = self.my_instrument.write(':SYST:PRES:TYPE FAC')
        
        # self.preset = self.my_instrument.write(':SYST:PRES')
        self._gpib_connection.write(':OUTPut:STATe OFF') # switch off the output
        self._gpib_connection.write('*RST')

        self.rf_info_msg.emit(repr(instrument_info))
        
    def rf_ply_stp(self):
        output_status = self._gpib_connection.query(':OUTPut:STATe?')
        
        if output_status == '0\n':
            frequency = float(self.cw_freq_spbx.value())*1e6
            power = float(self.cw_power_spbx.value())
            self.rf_ply_stp_btn.setText('RF OFF')
            self.off_icon = QIcon()
            self.off_icon.addPixmap(QPixmap(":/my_icons/images/icons/stop.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.off_icon)
            self._gpib_connection.write(':FREQ:MODE CW')
            self._gpib_connection.write(':FREQ:CW {0:f} Hz'.format(frequency))
            self._gpib_connection.write(':POWer:AMPLitude {0:f}'.format(power))
            rtn = self._gpib_connection.write(':OUTPut:STATe ON')
            if rtn != 0:
                self.rf_info_msg.emit('RF ON succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF ON failed')
                sys.emit()
        elif output_status == '1\n':
            self.rf_ply_stp_btn.setText('RF ON  ')
            self.on_icon = QIcon()
            self.on_icon.addPixmap(QPixmap(":/my_icons/images/icons/play.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.on_icon)
            rtn = self._gpib_connection.write(':OUTPut:STATe OFF')
            if rtn != 0:
                self.rf_info_msg.emit('RF OFF succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF OFF failed')
                sys.emit()
    def closeEvent(self, event):
        self._gpib_connection.write(':OUTPut:STATe OFF')
        self._gpib_connection.close()  
        self.rm.close()
        self.pulsestreamer_on_deactivate()
        self.timetagger_on_deactivate()
        return
if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec()
