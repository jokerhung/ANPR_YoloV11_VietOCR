import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature, decode_dss_signature
import struct
import platform
import os
import subprocess
import hashlib
import json
import time
from typing import Dict, Any, Optional
from mymachine import get_machine_key
import sys

class MyLicence:
    def __init__(self, uid: str, end: datetime):
        self.uid = uid
        self.end = end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "end": self.end.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MyLicence':
        end_time = datetime.fromisoformat(data["end"])
        return cls(uid=data["uid"], end=end_time)


class License:
    def __init__(self, data: bytes, signature: bytes):
        self.data = data
        self.signature = signature

    def verify(self, public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verify the license signature with the public key"""
        try:
            public_key.verify(self.signature, self.data)
            return True
        except Exception:
            return False

    def to_b32_string(self) -> str:
        """Convert license to base32 string"""
        # Combine data length, data, and signature
        data_length = len(self.data)
        combined = struct.pack('<I', data_length) + self.data + self.signature
        return base64.b32encode(combined).decode('ascii')

    @classmethod
    def from_b32_string(cls, license_b32: str) -> 'License':
        """Create license from base32 string"""
        try:
            combined = base64.b32decode(license_b32.encode('ascii'))
            
            # Extract data length
            data_length = struct.unpack('<I', combined[:4])[0]
            
            # Extract data and signature
            data = combined[4:4+data_length]
            signature = combined[4+data_length:]
            
            return cls(data=data, signature=signature)
        except Exception as e:
            raise ValueError(f"Invalid license format: {e}")


def private_key_from_b32_string(private_key_b32: str) -> ed25519.Ed25519PrivateKey:
    """Convert base32 string to Ed25519 private key"""
    try:
        key_bytes = base64.b32decode(private_key_b32.encode('ascii'))
        return ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid private key format: {e}")


def public_key_from_b32_string(public_key_b32: str) -> ed25519.Ed25519PublicKey:
    """Convert base32 string to Ed25519 public key"""
    try:
        key_bytes = base64.b32decode(public_key_b32.encode('ascii'))
        return ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid public key format: {e}")


def new_license(private_key: ed25519.Ed25519PrivateKey, data: bytes) -> License:
    """Create a new license with private key and data"""
    signature = private_key.sign(data)
    return License(data=data, signature=signature)


def generate_key_pair() -> tuple[str, str]:
    """Generate a new Ed25519 key pair and return as base32 strings"""
    #private_key = ed25519.Ed25519PrivateKey.generate()
    #public_key = private_key.public_key()
    
    # private_bytes = private_key.private_bytes(
    #     encoding=serialization.Encoding.Raw,
    #     format=serialization.PrivateFormat.Raw,
    #     encryption_algorithm=serialization.NoEncryption()
    # )
    
    # public_bytes = public_key.public_bytes(
    #     encoding=serialization.Encoding.Raw,
    #     format=serialization.PublicFormat.Raw
    # )
    
    # private_key_b32 = base64.b32encode(private_bytes).decode('ascii')
    # public_key_b32 = base64.b32encode(public_bytes).decode('ascii')

    private_key_b32 = "TACK32PAGSFW7FWV67EFSVI7BXRQ7PMY7XKHGDFBFDXHXNN4QXOQ===="
    public_key_b32 = "LFXNEKLMYOZ2PHS4WHUIKO7MGD6OTG6GRHLVDQ5C5K3LD7ANYDVA===="
    
    return private_key_b32, public_key_b32


def gen_lic(uid: str, private_key_base32: str) -> str:
    """Generate a license for the given UID using the private key"""
    try:
        # Unmarshal the private key
        private_key = private_key_from_b32_string(private_key_base32)
        
        # Define the data you need in your license
        # Here we use a struct that is marshalled to json
        doc = MyLicence(
            uid=uid,
            end=datetime.now() + timedelta(days=365 * 100)  # 100 years
        )
        
        # Marshall the document to bytes
        doc_bytes = json.dumps(doc.to_dict()).encode('utf-8')
        
        # Generate your license with the private key and the document
        license_obj = new_license(private_key, doc_bytes)
        
        # The b32 representation of our license
        license_b32 = license_obj.to_b32_string()
        
        return license_b32
        
    except Exception as e:
        raise Exception(f"Failed to generate license: {e}")


def verify_lic(license_b32: str, public_key_base32: str, uid: str) -> None:
    """Verify a license with the given public key and UID"""
    try:
        # Unmarshal the public key
        public_key = public_key_from_b32_string(public_key_base32)
        
        # Unmarshal the customer license
        license_obj = License.from_b32_string(license_b32)
        
        # Validate the license signature
        if not license_obj.verify(public_key):
            raise Exception("Invalid license signature")
        
        # Unmarshal the document
        doc_data = json.loads(license_obj.data.decode('utf-8'))
        result = MyLicence.from_dict(doc_data)
        
        # Check that the end date is after current time
        if result.end < datetime.now():
            raise Exception("License expired")
        
        # Check if the uid matches
        if result.uid != uid:
            raise Exception("License uid does not match")
            
    except Exception as e:
        raise Exception(f"License verification failed: {e}")

# Example usage and testing functions
def main():
    """Example usage of the license system"""
    # Generate a key pair
    private_key_b32, public_key_b32 = generate_key_pair()
    print(f"Private Key: {private_key_b32}")
    print(f"Public Key: {public_key_b32}")
    
    # Generate a license
    uid = input("Enter uid: ").strip()
    if not uid:
        print("Missing uid parameter")
        exit(1)
    license_b32 = gen_lic(uid, private_key_b32)
    print(f"uid: {uid}")
    print(f"License: {license_b32}")
    
    # Verify the license
    try:
        verify_lic(license_b32, public_key_b32, uid)
        print("License verification successful!")
    except Exception as e:
        print(f"License verification failed: {e}")

    # save licence key to file
    with open("licence.key", "w") as f:
        f.write(license_b32)

if __name__ == "__main__":
    main()