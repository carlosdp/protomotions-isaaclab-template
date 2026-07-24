# SPDX-FileCopyrightText: Copyright (c) 2026 Carlos Perez
# SPDX-License-Identifier: Apache-2.0
"""Check the host interfaces required by fully headless Isaac Lab."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import subprocess
import sys

VK_SUCCESS = 0
VK_API_VERSION_1_1 = (1 << 22) | (1 << 12)


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def check_platform() -> None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        fail(
            "this lock targets Linux x86_64; "
            f"detected {sys.platform} {platform.machine()}"
        )
    print("PASS: Linux x86_64", flush=True)


def check_driver_capabilities() -> None:
    value = os.environ.get("NVIDIA_DRIVER_CAPABILITIES")
    if value is None:
        print(
            "INFO: NVIDIA_DRIVER_CAPABILITIES is unset; "
            "the Vulkan probe will test the actual runtime",
            flush=True,
        )
        return

    capabilities = {item.strip() for item in value.split(",")}
    if "all" not in capabilities and "graphics" not in capabilities:
        print(
            "WARNING: NVIDIA_DRIVER_CAPABILITIES lacks 'graphics'. Set "
            "'compute,utility,graphics' when creating the container; "
            "exporting it after launch is not sufficient. Testing the actual "
            "Vulkan runtime next.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"PASS: NVIDIA driver capabilities include graphics ({value})",
            flush=True,
        )


def check_cuda_device() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        fail("nvidia-smi could not enumerate a GPU")

    devices = result.stdout.strip()
    if not devices:
        fail("nvidia-smi returned no GPUs")
    print(f"PASS: {devices}", flush=True)


def check_vulkan_instance() -> None:
    vulkan = None
    load_errors = []
    for library_name in filter(
        None, (ctypes.util.find_library("vulkan"), "libvulkan.so.1")
    ):
        try:
            vulkan = ctypes.CDLL(library_name)
            break
        except OSError as error:
            load_errors.append(str(error))

    if vulkan is None:
        fail(
            "the Vulkan loader is missing. Install libvulkan1 and vulkan-tools "
            f"in the container image. Loader errors: {'; '.join(load_errors)}"
        )

    vulkan.vkCreateInstance.argtypes = [
        ctypes.POINTER(VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    vulkan.vkCreateInstance.restype = ctypes.c_int32

    app_info = VkApplicationInfo(
        sType=0,
        pApplicationName=b"ProtoMotions Isaac Lab host check",
        applicationVersion=1,
        pEngineName=b"none",
        engineVersion=1,
        apiVersion=VK_API_VERSION_1_1,
    )
    create_info = VkInstanceCreateInfo(
        sType=1, pApplicationInfo=ctypes.pointer(app_info)
    )
    instance = ctypes.c_void_p()
    result = vulkan.vkCreateInstance(
        ctypes.byref(create_info), None, ctypes.byref(instance)
    )
    if result != VK_SUCCESS:
        fail(
            f"vkCreateInstance returned VkResult {result}. Verify the host "
            "driver and create the container with the NVIDIA graphics capability"
        )

    vulkan.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vulkan.vkDestroyInstance.restype = None
    vulkan.vkDestroyInstance(instance, None)
    print("PASS: Vulkan 1.1 instance creation", flush=True)


def main() -> None:
    check_platform()
    check_driver_capabilities()
    check_cuda_device()
    check_vulkan_instance()
    print(
        "Host preflight passed: fully headless Isaac Lab can initialize its GPU.",
        flush=True,
    )


if __name__ == "__main__":
    main()
