import os
import time
#import JSON-RPC Pulse Streamer wrapper class, to use Google-RPC import from pulsestreamer.grpc
from pulsestreamer import PulseStreamer, Sequence, OutputState, findPulseStreamers
# impofr timetagger 
os.environ['TIMETAGGER_INSTALL_PATH'] = 'C:\Program Files\Swabian Instruments\Time Tagger'
import TimeTagger as tt


# activate pulser
devices = findPulseStreamers()
# DHCP is activated in factory settings
if devices !=[]:
    ip = devices[0][0]
else:
    # if discovery failed try to connect by the default hostname
    # IP address of the pulse streamer (default hostname is 'pulsestreamer')
    print("No Pulse Streamer found")

#connect to the pulse streamer
pulser = PulseStreamer(ip)

# Print serial number and FPGA-ID
print('Serial: ' + pulser.getSerial())
print('FPGA ID: ' + pulser.getFPGAID())

'''
Set pulser and tagger
'''
ch_sig = 0
ch_gate = 1
ch_trigger=2

seq_gate = []
seq_sig = []

HIGH=1
LOW=0
for i in range(1,11):
    seq_gate += [(1500,HIGH),(500,LOW)]
    seq_sig += [(100,LOW)] + i*[(50,HIGH),(50,LOW)] + [(2000-100-i*100, LOW)]


#create the sequence
seq = Sequence()

#set digital channels
seq.setDigital(ch_gate, seq_gate)
seq.setDigital(ch_sig, seq_sig)
# seq.setDigital(ch_trigger, seq_trigger)

seq.plot()


final = OutputState.ZERO()
pulser.stream(seq, -1, final)



