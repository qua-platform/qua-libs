from QDAC_II import QDACII


# q = QDACII(visa_addr = "TCPIP::172.16.2.120::5025::SOCKET", lib = '@py')

q = QDACII()
print(q.query('*IDN?'))
print(q.query("syst:err:all?"))
print(q.query('syst:comm:lan:ipad?'))

channels = [7, 8]
for ch in channels:
# play sine tone
    q.write(f"sour{ch}:sine:freq 30000")
    q.write(f"sour{ch}:sine:span 1")
    q.write(f"sour{ch}:sine:count inf")
    q.write(f"sour{ch}:sine:trig:sour IMM")
    q.write(f"sour{ch}:sine:init")



