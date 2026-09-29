from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import os

import numpy as np


@dataclass(frozen=True)
class VirtualScreenGeometry:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureRegion:
    """Region in coordinates relative to the virtual-screen image."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


if os.name == "nt":
    SRCCOPY = 0x00CC0020
    CAPTUREBLT = 0x40000000
    DIB_RGB_COLORS = 0
    BI_RGB = 0
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class RGBQUAD(ctypes.Structure):
        _fields_ = [
            ("rgbBlue", ctypes.c_ubyte),
            ("rgbGreen", ctypes.c_ubyte),
            ("rgbRed", ctypes.c_ubyte),
            ("rgbReserved", ctypes.c_ubyte),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", RGBQUAD * 1),
        ]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    _user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    _user32.GetSystemMetrics.restype = ctypes.c_int
    _user32.GetDC.argtypes = (wintypes.HWND,)
    _user32.GetDC.restype = wintypes.HDC
    _user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    _user32.ReleaseDC.restype = ctypes.c_int

    _gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
    _gdi32.CreateCompatibleDC.restype = wintypes.HDC
    _gdi32.DeleteDC.argtypes = (wintypes.HDC,)
    _gdi32.DeleteDC.restype = wintypes.BOOL
    _gdi32.CreateCompatibleBitmap.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int)
    _gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
    _gdi32.SelectObject.restype = wintypes.HGDIOBJ
    _gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    _gdi32.DeleteObject.restype = wintypes.BOOL
    _gdi32.BitBlt.argtypes = (
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    )
    _gdi32.BitBlt.restype = wintypes.BOOL
    _gdi32.GetDIBits.argtypes = (
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(BITMAPINFO),
        wintypes.UINT,
    )
    _gdi32.GetDIBits.restype = ctypes.c_int
else:
    BITMAPINFOHEADER = object
    RGBQUAD = object
    BITMAPINFO = object
    _user32 = None
    _gdi32 = None


def _raise_last_error(operation: str) -> None:
    code = ctypes.get_last_error()
    raise OSError(code, f"{operation} failed")


def enable_dpi_awareness() -> None:
    """Best-effort process DPI awareness for pixel-exact desktop capture."""
    if os.name != "nt" or _user32 is None:
        return
    try:
        fn = _user32.SetProcessDpiAwarenessContext
        fn.argtypes = (ctypes.c_void_p,)
        fn.restype = wintypes.BOOL
        mask = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
        fn(ctypes.c_void_p(-4 & mask))
        return
    except (AttributeError, OSError, ValueError):
        pass
    try:
        fn = _user32.SetProcessDPIAware
        fn.argtypes = ()
        fn.restype = wintypes.BOOL
        fn()
    except (AttributeError, OSError):
        pass


def virtual_screen_geometry() -> VirtualScreenGeometry:
    if os.name != "nt" or _user32 is None:
        raise RuntimeError("Windows virtual-screen capture requires Windows")
    enable_dpi_awareness()
    x = int(_user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    y = int(_user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    width = int(_user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    height = int(_user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid virtual screen geometry: {(x, y, width, height)}")
    return VirtualScreenGeometry(x=x, y=y, width=width, height=height)


def clamp_capture_region(
    region: CaptureRegion,
    geometry: VirtualScreenGeometry,
) -> CaptureRegion:
    """Clamp a virtual-screen-relative region to the available desktop."""
    x1 = max(0, min(int(geometry.width), int(region.x)))
    y1 = max(0, min(int(geometry.height), int(region.y)))
    x2 = max(x1, min(int(geometry.width), int(region.x + region.width)))
    y2 = max(y1, min(int(geometry.height), int(region.y + region.height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Capture region is outside the virtual screen: {region}")
    return CaptureRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


class WindowsGdiScreenCapture:
    """Reusable Win32 GDI desktop capture.

    V1 allocated a DC, bitmap and pixel buffer on every frame.  The live vision
    loop now keeps those objects alive and only reallocates when the requested
    capture size changes.  ``grab_region`` additionally lets the temporal gate
    process just SELF + HOLD + NEXT after the one-time full-screen layout lock.
    """

    def __init__(self) -> None:
        if os.name != "nt" or _user32 is None or _gdi32 is None:
            raise RuntimeError("WindowsGdiScreenCapture requires Windows")
        enable_dpi_awareness()
        self.geometry = virtual_screen_geometry()
        self._screen_dc = _user32.GetDC(None)
        if not self._screen_dc:
            _raise_last_error("GetDC")
        self._memory_dc = _gdi32.CreateCompatibleDC(self._screen_dc)
        if not self._memory_dc:
            _user32.ReleaseDC(None, self._screen_dc)
            self._screen_dc = None
            _raise_last_error("CreateCompatibleDC")

        self._bitmap = None
        self._old_object = None
        self._surface_size: tuple[int, int] | None = None
        self._info = None
        self._raw = None
        self._closed = False

    def _destroy_surface(self) -> None:
        if self._old_object and self._memory_dc:
            _gdi32.SelectObject(self._memory_dc, self._old_object)
        self._old_object = None
        if self._bitmap:
            _gdi32.DeleteObject(self._bitmap)
        self._bitmap = None
        self._surface_size = None
        self._info = None
        self._raw = None

    def _ensure_surface(self, width: int, height: int) -> None:
        size = (int(width), int(height))
        if self._surface_size == size and self._bitmap and self._raw is not None:
            return
        self._destroy_surface()

        bitmap = _gdi32.CreateCompatibleBitmap(self._screen_dc, size[0], size[1])
        if not bitmap:
            _raise_last_error("CreateCompatibleBitmap")
        old_object = _gdi32.SelectObject(self._memory_dc, bitmap)
        if not old_object:
            _gdi32.DeleteObject(bitmap)
            _raise_last_error("SelectObject")

        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = size[0]
        info.bmiHeader.biHeight = -size[1]
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        info.bmiHeader.biSizeImage = size[0] * size[1] * 4

        self._bitmap = bitmap
        self._old_object = old_object
        self._surface_size = size
        self._info = info
        self._raw = (ctypes.c_ubyte * info.bmiHeader.biSizeImage)()

    def grab_region(self, region: CaptureRegion) -> np.ndarray:
        if self._closed:
            raise RuntimeError("WindowsGdiScreenCapture is closed")
        region = clamp_capture_region(region, self.geometry)
        self._ensure_surface(region.width, region.height)
        assert self._bitmap is not None
        assert self._info is not None
        assert self._raw is not None

        ok = _gdi32.BitBlt(
            self._memory_dc,
            0,
            0,
            region.width,
            region.height,
            self._screen_dc,
            self.geometry.x + region.x,
            self.geometry.y + region.y,
            SRCCOPY | CAPTUREBLT,
        )
        if not ok:
            _raise_last_error("BitBlt")

        scanlines = _gdi32.GetDIBits(
            self._memory_dc,
            self._bitmap,
            0,
            region.height,
            ctypes.byref(self._raw),
            ctypes.byref(self._info),
            DIB_RGB_COLORS,
        )
        if scanlines != region.height:
            _raise_last_error("GetDIBits")

        bgra = np.frombuffer(self._raw, dtype=np.uint8).reshape(
            region.height,
            region.width,
            4,
        )
        # Copy because the backing GDI buffer is reused on the next frame.
        return np.ascontiguousarray(bgra[:, :, :3])

    def grab(self) -> np.ndarray:
        return self.grab_region(
            CaptureRegion(0, 0, self.geometry.width, self.geometry.height)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._destroy_surface()
        if self._memory_dc:
            _gdi32.DeleteDC(self._memory_dc)
            self._memory_dc = None
        if self._screen_dc:
            _user32.ReleaseDC(None, self._screen_dc)
            self._screen_dc = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
