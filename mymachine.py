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

    try:
        # Execute the wmic command to get the UUID
        cmd = ["wmic", "csproduct", "get", "uuid"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Process the output
        lines = result.stdout.strip().split('\n')
        uuid_value = ""
        
        for line in lines:
            trimmed_line = line.strip()
            if trimmed_line and trimmed_line.upper() != "UUID":
                uuid_value = trimmed_line
                break
        
        if not uuid_value:
            raise Exception("UUID could not be retrieved")
            
        return uuid_value
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error executing command: {e}")
    except Exception as e:
        raise Exception(f"Error reading CPU ID: {e}")


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
    
    # Create raw machine key for debugging
    raw_machine_key = f"{cpu_id}-{machine_name}"
    
    return hash_string(raw_machine_key)