"""Before the lab: checks that the installed Hikrobot MVS SDK has everything camera/hikrobot_camera.py uses.
No camera needed.      python tests/check_mvs_sdk.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from camera.hikrobot_camera import load_sdk, mvimport_dirs  # noqa: E402

METHODS = ["MV_CC_EnumDevices", "MV_CC_CreateHandle", "MV_CC_OpenDevice", "MV_CC_SetEnumValue", "MV_CC_SetEnumValueByString",
           "MV_CC_SetFloatValue", "MV_CC_GetFloatValue", "MV_CC_GetIntValue", "MV_CC_SetIntValue", "MV_CC_SetCommandValue",
           "MV_CC_StartGrabbing", "MV_CC_StopGrabbing", "MV_CC_GetImageBuffer", "MV_CC_FreeImageBuffer",
           "MV_CC_CloseDevice", "MV_CC_DestroyHandle"]
NAMES = ["MV_USB_DEVICE", "MV_GIGE_DEVICE", "MV_ACCESS_Exclusive", "MV_CC_DEVICE_INFO_LIST", "MV_CC_DEVICE_INFO",
         "MVCC_FLOATVALUE", "MVCC_INTVALUE", "MV_FRAME_OUT"]


def main():
    print("MvImport search folders:", mvimport_dirs())
    print("MVCAM_COMMON_RUNENV =", os.environ.get("MVCAM_COMMON_RUNENV"))
    mvs = load_sdk()
    print("[OK]   MvCameraControl_class imported from", os.path.dirname(mvs.__file__))
    bad = 0
    for m in METHODS:
        ok = hasattr(mvs.MvCamera, m)
        bad += not ok
        print(("[OK]  " if ok else "[MISS]"), "MvCamera." + m)
    for n in NAMES:
        ok = hasattr(mvs, n)
        bad += not ok
        print(("[OK]  " if ok else "[MISS]"), n)
    fields = {"MV_FRAME_OUT": ["pBufAddr", "stFrameInfo"], "MVCC_FLOATVALUE": ["fCurValue", "fMin", "fMax"],
              "MVCC_INTVALUE": ["nCurValue", "nMin", "nMax", "nInc"]}
    for struct, fs in fields.items():
        have = [f[0] for f in getattr(getattr(mvs, struct, None), "_fields_", [])]
        for f in fs:
            ok = f in have
            bad += not ok
            print(("[OK]  " if ok else "[MISS]"), f"{struct}.{f}")
    if hasattr(mvs.MvCamera, "MV_CC_GetSDKVersion"):
        v = mvs.MvCamera.MV_CC_GetSDKVersion()
        print(f"SDK version: {v >> 24 & 0xFF}.{v >> 16 & 0xFF}.{v >> 8 & 0xFF}.{v & 0xFF}")
    if hasattr(mvs.MvCamera, "MV_CC_Initialize"):
        mvs.MvCamera.MV_CC_Initialize()
    dl = mvs.MV_CC_DEVICE_INFO_LIST()
    ret = mvs.MvCamera.MV_CC_EnumDevices(mvs.MV_USB_DEVICE | mvs.MV_GIGE_DEVICE, dl)
    print(f"enumerate cameras: return 0x{ret & 0xFFFFFFFF:08X}, cameras found: {dl.nDeviceNum} (0 is fine without the camera)")
    print("SDK CHECK PASSED" if bad == 0 else f"SDK CHECK: {bad} item(s) missing - send me this output")


if __name__ == "__main__":
    main()
