import ctypes
import sys

CU_CTX_SCHED_BLOCKING_SYNC = 4


def use_blocking_cuda_sync():
    try:
        driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int()
        if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)):
            return
        for index in range(count.value):
            device = ctypes.c_int()
            if not driver.cuDeviceGet(ctypes.byref(device), index):
                driver.cuDevicePrimaryCtxSetFlags_v2(device, CU_CTX_SCHED_BLOCKING_SYNC)
    except (AttributeError, OSError):
        pass
