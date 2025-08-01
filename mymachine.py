import os
import subprocess
import hashlib
import json
import platform
import time
from typing import Dict, Any, Optional


def read_cpu_id() -> str:
    """Retrieves the UUID (CPU ID) of the machine on a Windows system"""
    if not is_windows():
        raise Exception("This method only works on Windows")

    # Method 1: Try wmic command
    try:
        result = subprocess.run(['wmic', 'cpu', 'get', 'ProcessorId'], 
                              capture_output=True, text=True, check=True, 
                              creationflags=subprocess.CREATE_NO_WINDOW)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1 and lines[1].strip():
            return lines[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    
    # Method 2: Try PowerShell
    try:
        result = subprocess.run(['powershell', '-Command', 
                               'Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty ProcessorId'], 
                              capture_output=True, text=True, check=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        output = result.stdout.strip()
        if output and output != 'ProcessorId':
            return output
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return None

def is_windows() -> bool:
    """Checks if the operating system is Windows"""
    return platform.system().lower() == "windows"


def read_computer_name() -> str:
    """Retrieves the computer name (hostname) of the machine"""
    computer_name = os.environ.get("COMPUTERNAME")
    if not computer_name:
        return "UNKNOWN"
    return computer_name


def hash_string(input_str: str) -> str:
    """Hashes the input string using SHA-256 algorithm"""
    hash_obj = hashlib.sha256()
    hash_obj.update(input_str.encode('utf-8'))
    return hash_obj.hexdigest()


def get_machine_key() -> str:
    """Generates a unique machine key using CPU ID and machine name"""
    cpu_id = read_cpu_id()
    machine_name = read_computer_name()

    # check if cpu_id or machine_name is None or UNKNOWN
    if cpu_id is None or machine_name is None or machine_name == "UNKNOWN":
        raise Exception("Unable to retrieve CPU ID or machine name for machine key generation.")
    
    # Create raw machine key for debugging
    raw_machine_key = f"{cpu_id}-{machine_name}"
    
    return hash_string(raw_machine_key)