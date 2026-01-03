"""
Test script for Flask API
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_default():
    """Test default calculation endpoint."""
    print("Testing /calculate/default endpoint...")
    response = requests.get(f"{BASE_URL}/calculate/default")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"T_junction: {data['results']['t_junction']:.6f} °C")
        print(f"R_total: {data['thermal_resistances']['r_total']:.6f} °C/W")
    else:
        print(f"Error: {response.text}")

def test_custom():
    """Test custom calculation endpoint."""
    print("\nTesting /calculate endpoint with custom parameters...")
    payload = {
        "tdp": 150,
        "t_ambient": 25,
        "air_velocity": 1.5
    }
    response = requests.post(f"{BASE_URL}/calculate", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"T_junction: {data['results']['t_junction']:.6f} °C")
        print(f"R_total: {data['thermal_resistances']['r_total']:.6f} °C/W")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("=" * 60)
    print("Flask API Test Script")
    print("=" * 60)
    
    # Run tests
    try:
        test_default()
        test_custom()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API. Make sure it's running:")
        print("  python3 flask_api.py")
    
    print("\n" + "=" * 60)

