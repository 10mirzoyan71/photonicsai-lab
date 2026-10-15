"""Simulated scientific CMOS camera with the interface of camera/base.py, looking at one of two scenes.

Signal chain per pixel and frame (all parameters are attributes, so a test can compare a measurement with the truth):
    electrons = Poisson( (P(t) * laser_flux + ambient_flux) * t_exp * PRNU + dark_current * t_exp ) + Normal(0, read_noise)
    electrons are clipped at the full well; DN = electrons / K(gain) + black_level, rounded and clipped to the bit depth;
    K(gain) = K0 / 10^(gain_dB / 20) electrons per DN.
scene="speckle": a fully developed speckle field (circular Gaussian field low-passed by a circular pupil) under a Gaussian
beam envelope. P(t) is the laser power: a slow drift (a few percent over minutes) times a 1 % shot-to-shot fluctuation.
scene="room": no laser; the bare sensor (or a defocused lens) sees a white surface under mains-powered room lights. The
flux is nearly uniform (mild vignetting), drifts slowly, and is modulated at flicker_hz (100 Hz for 50 Hz mains) with
relative depth flicker_depth. A frame integrates the light over its exposure from a random phase of the flicker, so the
frame mean fluctuates unless the exposure is a whole number of flicker periods.
The simulation does not wait for the exposure time.
"""
from __future__ import annotations

import time

import numpy as np

from .base import Camera, CameraError


class SimCamera(Camera):
    vendor = "simulation"
    model = "SimCam-640 (simulated 12-bit CMOS)"

    def __init__(self, seed: int = 1, width: int = 640, height: int = 512, bit_depth: int = 12,
                 full_well_e: float = 10000.0, k0_e_per_dn: float = 2.4, read_noise_e: float = 6.0,
                 black_level_dn: float = 40.0, dark_current_e_s: float = 25.0, prnu: float = 0.01,
                 hot_pixel_fraction: float = 5e-4, exposure_range_us=(20.0, 2_000_000.0), gain_range_db=(0.0, 24.0),
                 grain_px: float = 3.0, beam_waist_frac: float = 0.3, peak_flux_e_s: float = 4.0e6,
                 ambient_e_s: float = 2.0e3, laser_noise: float = 0.01, drift_amplitude: float = 0.03,
                 drift_period_s: float = 300.0, scene: str = "speckle", room_flux_e_s: float = 1.5e5,
                 flicker_hz: float = 100.0, flicker_depth: float = 0.3):
        if scene not in ("speckle", "room"):
            raise CameraError(f"unknown scene {scene!r} (use 'speckle' or 'room')")
        self.rng = np.random.default_rng(seed)
        self.scene = scene
        self.flicker_hz, self.flicker_depth = (flicker_hz, flicker_depth) if scene == "room" else (0.0, 0.0)
        self.width, self.height, self.bit_depth = width, height, bit_depth
        self.full_well_e, self.k0, self.read_noise_e = full_well_e, k0_e_per_dn, read_noise_e
        self.black_level_dn, self.dark_current_e_s = black_level_dn, dark_current_e_s
        self.exposure_range_us, self.gain_range_db = tuple(exposure_range_us), tuple(gain_range_db)
        self.ambient_e_s, self.laser_noise = ambient_e_s, laser_noise
        self.drift_amplitude, self.drift_period_s = drift_amplitude, drift_period_s
        self._drift_phase = self.rng.uniform(0, 2 * np.pi)
        self._t0 = time.monotonic()
        self._clock = None                     # tests can set a fake clock (seconds)

        # fixed-pattern maps
        self.prnu_map = 1.0 + prnu * self.rng.standard_normal((height, width))
        dark = np.full((height, width), dark_current_e_s)
        hot = self.rng.random((height, width)) < hot_pixel_fraction
        dark[hot] = dark_current_e_s * self.rng.uniform(50, 500, hot.sum())
        self.dark_map = dark

        # scene, flux in electrons per second per pixel
        if scene == "speckle":                 # speckle under a Gaussian envelope
            self.laser_flux = peak_flux_e_s * self._speckle(grain_px) * self._envelope(beam_waist_frac)
        else:                                  # white surface under room light: no laser, the light is the 'ambient' term
            self.laser_flux = np.zeros((height, width))
            self.ambient_e_s = room_flux_e_s * self._vignetting()
            self.laser_noise = 0.003           # the lamps' own shot-to-shot noise, besides the flicker

        self.exposure_us, self.gain_db = 1000.0, 0.0
        self.adc_bits = bit_depth              # an 8-bit ADC leaves 256 levels of the 12-bit output (steps of ~16 DN)
        self.roi = (0, 0, width, height)
        self.illumination = 1.0
        self.frames_grabbed = 0

    # -- scene ---------------------------------------------------------------------------
    def _speckle(self, grain_px):
        h, w = self.height, self.width
        fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
        pupil = np.hypot(fx, fy) <= 1.0 / (2.0 * grain_px)
        field = np.fft.ifft2(np.fft.fft2(self.rng.standard_normal((h, w)) + 1j * self.rng.standard_normal((h, w))) * pupil)
        inten = np.abs(field) ** 2
        return inten / inten.mean()

    def _envelope(self, waist_frac):
        h, w = self.height, self.width
        y, x = np.mgrid[0:h, 0:w]
        r2 = (x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2
        wpx = waist_frac * min(h, w)
        return np.exp(-2.0 * r2 / wpx ** 2)

    def _vignetting(self):
        h, w = self.height, self.width
        y, x = np.mgrid[0:h, 0:w]
        r2 = ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) / (0.5 * np.hypot(w, h)) ** 2
        return 1.0 - 0.15 * r2

    def flicker_factor(self, t_exp_s: float) -> float:
        """Mean of 1 + depth*cos(2 pi f t) over an exposure that starts at a random phase of the flicker."""
        if self.flicker_depth == 0.0:
            return 1.0
        w = 2 * np.pi * self.flicker_hz
        phi = self.rng.uniform(0, 2 * np.pi)
        return 1.0 + self.flicker_depth * (np.sin(w * t_exp_s + phi) - np.sin(phi)) / (w * t_exp_s)

    def _now(self):
        return self._clock if self._clock is not None else time.monotonic() - self._t0

    def laser_power(self) -> float:
        drift = 1.0 + self.drift_amplitude * np.sin(2 * np.pi * self._now() / self.drift_period_s + self._drift_phase)
        return drift * (1.0 + self.laser_noise * self.rng.standard_normal())

    def conversion_gain_e_per_dn(self, gain_db=None) -> float:
        g = self.gain_db if gain_db is None else gain_db
        return self.k0 / 10 ** (g / 20.0)

    # -- interface -------------------------------------------------------------------------
    def info(self) -> dict:
        return {"vendor": self.vendor, "model": self.model, "sensor_width": self.width, "sensor_height": self.height,
                "bit_depth": self.bit_depth, "max_dn": 2 ** self.bit_depth - 1,
                "exposure_range_us": list(self.exposure_range_us), "gain_range_db": list(self.gain_range_db),
                "exposure_us": self.exposure_us, "gain_db": self.gain_db,
                "roi": dict(zip(("x", "y", "width", "height"), self.roi)),
                "adc_bit_depth": self.adc_bits, "black_level_setting": self.black_level_dn,
                "illumination_control": True, "illumination": self.illumination, "simulated": True, "scene": self.scene}

    def set_exposure(self, exposure_us: float) -> float:
        lo, hi = self.exposure_range_us
        if not np.isfinite(exposure_us) or exposure_us < lo or exposure_us > hi:
            raise CameraError(f"exposure {exposure_us} us outside the camera range {lo}..{hi} us")
        self.exposure_us = float(round(exposure_us))              # 1 us resolution
        return self.exposure_us

    def set_gain(self, gain_db: float) -> float:
        lo, hi = self.gain_range_db
        if not np.isfinite(gain_db) or gain_db < lo or gain_db > hi:
            raise CameraError(f"gain {gain_db} dB outside the camera range {lo}..{hi} dB")
        self.gain_db = float(round(gain_db, 1))                   # 0.1 dB resolution
        return self.gain_db

    def set_roi(self, x: int, y: int, width: int, height: int):
        x, y, width, height = (int(v) for v in (x, y, width, height))
        if x < 0 or y < 0 or width < 16 or height < 16 or x + width > self.width or y + height > self.height:
            raise CameraError(f"ROI ({x}, {y}, {width}, {height}) does not fit the {self.width}x{self.height} sensor "
                              "(minimum 16x16)")
        self.roi = (x, y, width, height)
        return self.roi

    def set_black_level(self, value: int) -> int:
        """In the simulation the black level is the dark offset itself, in DN."""
        if not 0 <= int(value) <= 500:
            raise CameraError("black level must be 0..500 DN on the simulated camera")
        self.black_level_dn = float(int(value))
        return int(value)

    def set_adc_bit_depth(self, bits: int) -> int:
        if int(bits) not in (8, self.bit_depth):
            raise CameraError(f"ADC bit depth must be 8 or {self.bit_depth}")
        self.adc_bits = int(bits)
        return self.adc_bits

    def temperature_c(self) -> float:
        """A simulated sensor that warms from 38 C towards 41 C over about 10 minutes."""
        return round(38.0 + 3.0 * (1 - np.exp(-self._now() / 600.0)) + 0.05 * self.rng.standard_normal(), 2)

    def set_illumination(self, level: float) -> float:
        if not 0.0 <= level <= 1.0:
            raise CameraError("illumination level must be between 0 and 1")
        self.illumination = float(level)
        return self.illumination

    def grab(self) -> np.ndarray:
        x, y, w, h = self.roi
        sl = (slice(y, y + h), slice(x, x + w))
        t = self.exposure_us * 1e-6
        amb = self.ambient_e_s[sl] if np.ndim(self.ambient_e_s) else self.ambient_e_s
        if self.scene == "speckle":
            photo = self.illumination * self.laser_power() * self.laser_flux[sl] + amb * self.illumination
        else:                                  # room light: drift, lamp noise and flicker all act on the ambient light
            photo = self.illumination * self.laser_power() * self.flicker_factor(t) * amb
        mean_e = photo * t * self.prnu_map[sl] + self.dark_map[sl] * t
        e = self.rng.poisson(mean_e).astype(float) + self.read_noise_e * self.rng.standard_normal(mean_e.shape)
        e = np.minimum(e, self.full_well_e)                       # full well; read noise below zero survives the offset
        dn = np.round(e / self.conversion_gain_e_per_dn() + self.black_level_dn)
        if self.adc_bits < self.bit_depth:                        # coarser ADC: fewer levels across the same range
            step = (2 ** self.bit_depth - 1) / (2 ** self.adc_bits - 1)
            dn = np.round(np.round(dn / step) * step)
        self.frames_grabbed += 1
        return np.clip(dn, 0, 2 ** self.bit_depth - 1).astype(np.uint16)

    def truth(self) -> dict:
        """Ground-truth parameters, for checking measurements made through the agent (never shown to the agent)."""
        return {"conversion_gain_e_per_dn_at_0dB": self.k0, "read_noise_e": self.read_noise_e,
                "full_well_e": self.full_well_e, "black_level_dn": self.black_level_dn,
                "dark_current_e_s": self.dark_current_e_s,
                "read_noise_dn_at_0dB": self.read_noise_e / self.k0, "scene": self.scene,
                "flicker_hz": self.flicker_hz, "flicker_depth": self.flicker_depth}
