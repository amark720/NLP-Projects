import ctypes, os, traceback
path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cublas64_11.dll'
print('exists', os.path.exists(path))
print('path', path)
try:
    ctypes.WinDLL(path)
    print('OK loaded')
except Exception as e:
    print(type(e).__name__, e)
    traceback.print_exc()
