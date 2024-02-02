import struct
import numpy as np

def data_reader(d_path, RECORD_LENGTH, amount):

  print("	- Unpacking .dat file:")
  with open(d_path, 'rb') as f:
    data = f.read()

  # Header has  24 bytes
  h_size = 24
  # Waveforms have RECORD_LENGTH x 2bytes
  w_size = RECORD_LENGTH*2
  max_count = 200 # quantity of waveforms for the 1st analyses
  events = [] # List with header and waveforms of each event
  count = 0
  step = h_size + w_size
  wf = []
  count = 0
  for i in range(0,len(data), step):     # Selecting the indexes of each event start
    # To separate the waveforms from the headers:
      if i+(h_size-1) < len(data):
        event = data[i+h_size : i + step]
        events.append(event)

      # To convert bytes into integers
      tick = 0
      count2 = 1
      waveform = []
      for j in range(0,len(event),2):
        waveform_bin = event[j: j+2]
        waveform_int = (struct.unpack('h', waveform_bin)[0])	
        waveform.append(waveform_int)

      if amount != "all":
        if count < amount:
          wv = np.array(waveform)
          wf.append(wv)
          #plt.plot(wv)
          #plt.show()
          count += 1
          #print("		Waveforms recorded: ", count)
        else:
          break

      else:
        wv = np.array(waveform)
        wf.append(wv)
        if (i == 0.2*(len(data))) : print("		20% done ...")
        if (i == 0.4*(len(data))) : print("		40% done ...")
        if (i == 0.6*(len(data))) : print("		60% done ...")
        if (i == 0.8*(len(data))) : print("		80% done ...")
  print("		100% done!")
  return wf
