"""Connector to the lab camera: Hikrobot MV-CS028-10UM (monochrome, USB 3.0, USB3 Vision / GenICam), through Hikrobot's
MVS SDK (MvCameraControl library) and the Python wrapper that MVS installs (MvImport/MvCameraControl_class.py).

How it drives the camera (GenICam feature names, as in the MVS client's feature tree):
    open        MV_CC_EnumDevices -> MV_CC_CreateHandle -> MV_CC_OpenDevice (exclusive: close the MVS client first)
    setup       ExposureAuto=Off, GainAuto=Off, PixelFormat = Mono12 (else Mono10, else Mono8),
                TriggerMode=On, TriggerSource=Software, then MV_CC_StartGrabbing
    exposure    ExposureTime (float, microseconds)          gain   Gain (float, dB)
    ROI         OffsetX/OffsetY/Width/Height (integers, with the increments the camera reports; acquisition stopped)
    grab        TriggerSoftware command -> MV_CC_GetImageBuffer -> copy to numpy (uint16) -> MV_CC_FreeImageBuffer
                With a software trigger every frame is exposed AFTER the command, with the settings just applied.
Sensor size, bit depth and the exposure/gain ranges are read from the camera, not assumed.

The wrapper is found through MVS_MVIMPORT, or the default MVS install folders below.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from .base import Camera, CameraError

PIXEL_FORMATS = [("Mono12", 0x01100005, 12), ("Mono10", 0x01100003, 10), ("Mono8", 0x01080001, 8)]
RUNTIME_DIRS = [r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
                r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"]


def _machine_env(name):
    """A system environment variable as the MVS installer set it (programs started before the install miss it)."""
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            return os.path.expandvars(winreg.QueryValueEx(k, name)[0])
    except OSError:
        return None


def mvimport_dirs():
    runenv = os.environ.get("MVCAM_COMMON_RUNENV") or _machine_env("MVCAM_COMMON_RUNENV") or ""
    return [d for d in (os.environ.get("MVS_MVIMPORT", ""),
                        os.path.join(runenv, "Samples", "Python", "MvImport") if runenv else "",
                        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
                        r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
                        "/opt/MVS/Samples/64/Python/MvImport") if d]


def load_sdk():
    """Import the MVS Python wrapper, with its MvCameraControl.dll findable (the wrapper loads it by name)."""
    runenv = os.environ.get("MVCAM_COMMON_RUNENV") or _machine_env("MVCAM_COMMON_RUNENV")
    if runenv:
        os.environ["MVCAM_COMMON_RUNENV"] = runenv
    if os.name == "nt":
        dll_dirs = [d for d in (_machine_env("Path") or "").split(";") if "MVS" in d and "Runtime" in d] + RUNTIME_DIRS
        for d in dll_dirs:
            if os.path.isdir(d):
                if d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
    for d in mvimport_dirs():
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    try:
        import MvCameraControl_class as mvs  # noqa: F401  (from the MVS install)
    except Exception as e:                   # noqa: BLE001
        raise CameraError("Hikrobot MVS SDK not found: install MVS (Machine Vision Software) from Hikrobot, or set "
                          f"MVS_MVIMPORT to its Samples/Python/MvImport folder ({type(e).__name__}: {e})") from e
    return mvs


def _bits(name):
    """12 from an ADCBitDepth entry such as 'Bits_12' (the name itself if it has no number)."""
    digits = "".join(c for c in str(name) if c.isdigit())
    return int(digits) if digits else name


def _cstr(arr) -> str:
    return bytes(bytearray(arr)).split(b"\0", 1)[0].decode("ascii", "replace")


class HikrobotCamera(Camera):
    vendor = "Hikrobot"

    def __init__(self, index: int = 0, serial: str | None = None, pixel_format: str | None = None,
                 black_level: int | None = 180):
        mvs = self.mvs = load_sdk()
        if hasattr(mvs.MvCamera, "MV_CC_Initialize"):          # MVS SDK >= 4.1 needs it once per process
            mvs.MvCamera.MV_CC_Initialize()
        dl = mvs.MV_CC_DEVICE_INFO_LIST()
        self._check(mvs.MvCamera.MV_CC_EnumDevices(mvs.MV_USB_DEVICE | mvs.MV_GIGE_DEVICE, dl), "enumerate cameras")
        if dl.nDeviceNum == 0:
            raise CameraError("no Hikrobot camera found: is it on a USB 3 port, and is the MVS client closed?")
        devs = []
        for i in range(dl.nDeviceNum):
            info = ctypes.cast(dl.pDeviceInfo[i], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == mvs.MV_USB_DEVICE:
                u = info.SpecialInfo.stUsb3VInfo
                devs.append((info, _cstr(u.chModelName), _cstr(u.chSerialNumber)))
            else:
                g = info.SpecialInfo.stGigEInfo
                devs.append((info, _cstr(g.chModelName), _cstr(g.chSerialNumber)))
        pick = [d for d in devs if d[2] == serial] if serial else devs[index:index + 1]
        if not pick:
            raise CameraError(f"camera not found (serial {serial!r}, index {index}); found {[(m, s) for _, m, s in devs]}")
        info, self.model, self.serial = pick[0]
        self.cam = mvs.MvCamera()
        self._check(self.cam.MV_CC_CreateHandle(info), "create camera handle")
        self._check(self.cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0),
                    "open the camera (another program, e.g. the MVS client, may be using it)")
        self._opened, self._grabbing = True, False
        for node in ("ExposureAuto", "GainAuto"):
            self.cam.MV_CC_SetEnumValue(node, 0)               # Off; ignored if the camera has no such node
        self.adc_bit_depth = self._max_adc_bit_depth()
        # BlackLevel: on this camera a larger value lowers the dark level; the default 240 puts it at 0 DN, which clips
        # half of the dark noise. 180 keeps it about 60 DN above zero (measured at the lab, 12-bit ADC, 0 dB).
        self.black_level_setting = black_level
        if black_level is not None:
            self.cam.MV_CC_SetIntValueEx("BlackLevel", int(black_level))
        self._set_pixel_format(pixel_format)
        self._check(self.cam.MV_CC_SetEnumValue("TriggerMode", 1), "TriggerMode=On")
        self._check(self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Software"), "TriggerSource=Software")
        self._start()

    # -- low level -----------------------------------------------------------------------
    @staticmethod
    def _check(ret, what):
        if ret != 0:
            raise CameraError(f"{what} failed: MVS error 0x{ret & 0xFFFFFFFF:08X}")

    def _float(self, node):
        v = self.mvs.MVCC_FLOATVALUE()
        self._check(self.cam.MV_CC_GetFloatValue(node, v), f"read {node}")
        return v.fCurValue, v.fMin, v.fMax

    def _int(self, node):
        v = self.mvs.MVCC_INTVALUE()
        self._check(self.cam.MV_CC_GetIntValue(node, v), f"read {node}")
        return v.nCurValue, v.nMin, v.nMax, max(1, v.nInc)

    def _max_adc_bit_depth(self):
        """The MV-CS028-10UM can digitise with 8 or 12 bits (ADCBitDepth); in the 8-bit mode a Mono12 frame has only
        about 240 levels (steps of 17 DN). Use the deepest mode the camera offers; returns its name (None if no such node)."""
        v = self.mvs.MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue("ADCBitDepth", v) != 0:
            return None
        best = max(v.nSupportValue[i] for i in range(v.nSupportedNum))
        self._check(self.cam.MV_CC_SetEnumValue("ADCBitDepth", best), "set ADCBitDepth")
        e = self.mvs.MVCC_ENUMENTRY()
        e.nValue = best
        if self.cam.MV_CC_GetEnumEntrySymbolic("ADCBitDepth", e) != 0:
            return str(best)
        return bytes(e.chSymbolic).split(b"\0")[0].decode()

    def _set_pixel_format(self, wanted):
        for name, code, bits in PIXEL_FORMATS:
            if wanted and name != wanted:
                continue
            if self.cam.MV_CC_SetEnumValue("PixelFormat", code) == 0:
                self.pixel_format, self.bit_depth = name, bits
                return
        raise CameraError(f"none of the pixel formats {[p[0] for p in PIXEL_FORMATS]} could be set")

    def _start(self):
        if not self._grabbing:
            self._check(self.cam.MV_CC_StartGrabbing(), "start acquisition")
            self._grabbing = True

    def _stop(self):
        if self._grabbing:
            self.cam.MV_CC_StopGrabbing()
            self._grabbing = False

    # -- interface -------------------------------------------------------------------------
    def info(self) -> dict:
        e, emin, emax = self._float("ExposureTime")
        g, gmin, gmax = self._float("Gain")
        wmax = self._int("WidthMax")[0]
        hmax = self._int("HeightMax")[0]
        roi = {"x": self._int("OffsetX")[0], "y": self._int("OffsetY")[0], "width": self._int("Width")[0],
               "height": self._int("Height")[0]}
        return {"vendor": self.vendor, "model": self.model, "serial": self.serial, "interface": "USB3 Vision via Hikrobot MVS SDK",
                "sensor_width": wmax, "sensor_height": hmax, "bit_depth": self.bit_depth, "pixel_format": self.pixel_format,
                "adc_bit_depth": _bits(self.adc_bit_depth), "black_level_setting": self.black_level_setting,
                "max_dn": 2 ** self.bit_depth - 1, "exposure_range_us": [emin, emax], "gain_range_db": [gmin, gmax],
                "exposure_us": round(e, 2), "gain_db": round(g, 3), "roi": roi, "illumination_control": False,
                "simulated": False}

    def set_exposure(self, exposure_us: float) -> float:
        _, lo, hi = self._float("ExposureTime")
        if not lo <= exposure_us <= hi:
            raise CameraError(f"exposure {exposure_us} us outside the camera range {lo:g}..{hi:g} us")
        self._check(self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)), "set ExposureTime")
        return round(self._float("ExposureTime")[0], 2)

    def set_gain(self, gain_db: float) -> float:
        _, lo, hi = self._float("Gain")
        if not lo <= gain_db <= hi:
            raise CameraError(f"gain {gain_db} dB outside the camera range {lo:g}..{hi:g} dB")
        self._check(self.cam.MV_CC_SetFloatValue("Gain", float(gain_db)), "set Gain")
        return round(self._float("Gain")[0], 3)

    def set_roi(self, x: int, y: int, width: int, height: int):
        wmax, hmax = self._int("WidthMax")[0], self._int("HeightMax")[0]
        if x < 0 or y < 0 or x + width > wmax or y + height > hmax:
            raise CameraError(f"ROI ({x}, {y}, {width}, {height}) does not fit the {wmax}x{hmax} sensor")
        self._stop()
        try:
            self.cam.MV_CC_SetIntValue("OffsetX", 0)
            self.cam.MV_CC_SetIntValue("OffsetY", 0)
            for node, val in (("Width", width), ("Height", height), ("OffsetX", x), ("OffsetY", y)):
                _, lo, hi, inc = self._int(node)
                v = int(min(max(lo, lo + round((val - lo) / inc) * inc), hi))   # nearest allowed value
                self._check(self.cam.MV_CC_SetIntValue(node, v), f"set {node}={v}")
        finally:
            self._start()
        r = self.info()["roi"]
        return r["x"], r["y"], r["width"], r["height"]

    def set_black_level(self, value: int) -> int:
        """BlackLevel (integer node). On this camera a LARGER value gives a LOWER dark level: 240 puts it at 0 DN (half of
        the dark noise clipped), 180 at about 62 DN (lab, 12-bit ADC, 0 dB)."""
        _, lo, hi, _ = self._int("BlackLevel")
        if not lo <= int(value) <= hi:
            raise CameraError(f"black level {value} outside the camera range {lo}..{hi}")
        self._check(self.cam.MV_CC_SetIntValueEx("BlackLevel", int(value)), "set BlackLevel")
        self.black_level_setting = int(self._int("BlackLevel")[0])
        return self.black_level_setting

    def set_adc_bit_depth(self, bits: int) -> int:
        """ADCBitDepth (enumeration such as Bits_8 / Bits_12), changed with the acquisition stopped."""
        v = self.mvs.MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue("ADCBitDepth", v) != 0:
            raise CameraError("this camera has no ADCBitDepth setting")
        offered = {}
        for i in range(v.nSupportedNum):
            e = self.mvs.MVCC_ENUMENTRY()
            e.nValue = v.nSupportValue[i]
            name = (bytes(e.chSymbolic).split(b"\0")[0].decode()
                    if self.cam.MV_CC_GetEnumEntrySymbolic("ADCBitDepth", e) == 0 else str(v.nSupportValue[i]))
            offered[_bits(name)] = (v.nSupportValue[i], name)
        if int(bits) not in offered:
            raise CameraError(f"ADC bit depth {bits} is not offered; the camera offers {sorted(offered, key=str)}")
        self._stop()
        try:
            self._check(self.cam.MV_CC_SetEnumValue("ADCBitDepth", offered[int(bits)][0]), "set ADCBitDepth")
        finally:
            self._start()
        self.adc_bit_depth = offered[int(bits)][1]
        return int(bits)

    def temperature_c(self) -> float:
        return round(float(self._float("DeviceTemperature")[0]), 2)

    def grab(self) -> np.ndarray:
        self._start()
        self._check(self.cam.MV_CC_SetCommandValue("TriggerSoftware"), "software trigger")
        frame = self.mvs.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
        timeout_ms = int(self._float("ExposureTime")[0] / 1000) + 2000
        self._check(self.cam.MV_CC_GetImageBuffer(frame, timeout_ms), "get frame")
        try:
            fi = frame.stFrameInfo
            raw = ctypes.string_at(frame.pBufAddr, fi.nFrameLen)
            if self.bit_depth == 8:
                img = np.frombuffer(raw, np.uint8)[: fi.nWidth * fi.nHeight].astype(np.uint16)
            else:                                              # Mono10/Mono12 unpacked: 2 bytes per pixel
                img = np.frombuffer(raw, "<u2")[: fi.nWidth * fi.nHeight].copy()
            return img.reshape(fi.nHeight, fi.nWidth)
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def close(self):
        if getattr(self, "_opened", False):
            self._stop()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self._opened = False
            if hasattr(self.mvs.MvCamera, "MV_CC_Finalize"):
                self.mvs.MvCamera.MV_CC_Finalize()
