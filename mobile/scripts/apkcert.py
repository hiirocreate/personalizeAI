# APK (v2/v3 署名) の証明書 SHA-256 を表示する
import struct, hashlib, sys
d=open(sys.argv[1],'rb').read()
m=d.rfind(b'APK Sig Block 42'); size=struct.unpack('<Q',d[m-8:m])[0]
start=m+16-size-8; blk=d[start+8:m-8]; i=0
def lp(b,o): n=struct.unpack('<I',b[o:o+4])[0]; return b[o+4:o+4+n], o+4+n
while i<len(blk):
    ln=struct.unpack('<Q',blk[i:i+8])[0]; bid=struct.unpack('<I',blk[i+8:i+12])[0]; val=blk[i+12:i+8+ln]; i+=8+ln
    if bid in (0x7109871a,0xf05368c0):
        signers,_=lp(val,0); signer,_=lp(signers,0); sd,_=lp(signer,0)
        digests,o=lp(sd,0); certs,o=lp(sd,o); cert,_=lp(certs,0)
        print(hex(bid), hashlib.sha256(cert).hexdigest().upper()); break
