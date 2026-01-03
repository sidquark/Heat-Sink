"""
Flask API for Thermal Model
Provides REST endpoints for heat sink thermal calculations
"""

from flask import Flask, request, jsonify
from thermal_model import ThermalModel
import json

app = Flask(__name__)


@app.route('/')
def index():
    """API documentation endpoint."""
    return jsonify({
        "message": "Thermal Model API",
        "version": "1.0",
        "endpoints": {
            "/": "API documentation",
            "/calculate": "Calculate thermal performance (POST)",
            "/calculate/default": "Calculate with default parameters (GET)",
            "/health": "Health check endpoint"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route('/calculate/default', methods=['GET'])
def calculate_default():
    """
    Calculate thermal performance with default parameters.
    
    Returns:
        JSON: Complete thermal analysis results
    """
    try:
        model = ThermalModel()
        results = model.calculate_junction_temperature()
        
        # Format results for JSON response
        response = {
            "success": True,
            "input_parameters": {
                "tdp": model.tdp,
                "t_ambient": model.t_ambient,
                "air_velocity": model.air_velocity
            },
            "thermal_resistances": {
                "r_jc": results['r_jc'],
                "r_tim": results['r_tim'],
                "r_cond": results['r_cond'],
                "r_conv": results['r_conv'],
                "r_hs": results['r_hs'],
                "r_total": results['r_total']
            },
            "convection_details": {
                "reynolds_number": results['reynolds'],
                "nusselt_number": results['nusselt'],
                "heat_transfer_coefficient": results['h_coefficient'],
                "convection_area": results['convection_area']
            },
            "results": {
                "t_ambient": results['t_ambient'],
                "t_junction": results['t_junction'],
                "tdp": results['tdp']
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Calculate thermal performance with custom parameters.
    
    Request Body (JSON):
    {
        "tdp": 150,                    // Thermal Design Power (W)
        "t_ambient": 25,               // Ambient temperature (°C)
        "air_velocity": 1,             // Air velocity (m/s)
        "r_jc": 0.2,                   // Junction-to-case resistance (°C/W) [optional]
        "die_length": 0.0525,          // Die length (m) [optional]
        "die_width": 0.045,            // Die width (m) [optional]
        "num_fins": 60,                // Number of fins [optional]
        "fin_height": 0.0245,          // Fin height (m) [optional]
        "fin_thickness": 0.0008,        // Fin thickness (m) [optional]
        "sink_length": 0.09,           // Sink length (m) [optional]
        "sink_width": 0.116            // Sink width (m) [optional]
    }
    
    Returns:
        JSON: Complete thermal analysis results
    """
    try:
        data = request.get_json() or {}
        
        # Create model with default values
        model = ThermalModel()
        
        # Update parameters if provided
        if 'tdp' in data:
            model.tdp = float(data['tdp'])
        if 't_ambient' in data:
            model.t_ambient = float(data['t_ambient'])
        if 'air_velocity' in data:
            model.air_velocity = float(data['air_velocity'])
        if 'r_jc' in data:
            model.r_jc = float(data['r_jc'])
        
        # Update geometry if provided
        if 'die_length' in data:
            model.die_length = float(data['die_length'])
            model.die_area = model.die_length * model.die_width
        if 'die_width' in data:
            model.die_width = float(data['die_width'])
            model.die_area = model.die_length * model.die_width
        if 'num_fins' in data:
            model.num_fins = int(data['num_fins'])
            model.fin_spacing = (model.sink_width - (model.num_fins * model.fin_thickness)) / (model.num_fins - 1)
        if 'fin_height' in data:
            model.fin_height = float(data['fin_height'])
        if 'fin_thickness' in data:
            model.fin_thickness = float(data['fin_thickness'])
            model.fin_spacing = (model.sink_width - (model.num_fins * model.fin_thickness)) / (model.num_fins - 1)
        if 'sink_length' in data:
            model.sink_length = float(data['sink_length'])
            model.base_area = model.sink_length * model.sink_width
        if 'sink_width' in data:
            model.sink_width = float(data['sink_width'])
            model.base_area = model.sink_length * model.sink_width
            model.fin_spacing = (model.sink_width - (model.num_fins * model.fin_thickness)) / (model.num_fins - 1)
        
        # Calculate results
        results = model.calculate_junction_temperature()
        
        # Format response
        response = {
            "success": True,
            "input_parameters": {
                "tdp": model.tdp,
                "t_ambient": model.t_ambient,
                "air_velocity": model.air_velocity,
                "r_jc": model.r_jc
            },
            "thermal_resistances": {
                "r_jc": results['r_jc'],
                "r_tim": results['r_tim'],
                "r_cond": results['r_cond'],
                "r_conv": results['r_conv'],
                "r_hs": results['r_hs'],
                "r_total": results['r_total']
            },
            "convection_details": {
                "reynolds_number": results['reynolds'],
                "nusselt_number": results['nusselt'],
                "heat_transfer_coefficient": results['h_coefficient'],
                "convection_area": results['convection_area']
            },
            "results": {
                "t_ambient": results['t_ambient'],
                "t_junction": results['t_junction'],
                "tdp": results['tdp']
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid parameter value: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("Starting Thermal Model API...")
    print("API Documentation: http://localhost:5000/")
    print("Default calculation: http://localhost:5000/calculate/default")
    print("Custom calculation: POST to http://localhost:5000/calculate")
    app.run(debug=True, host='0.0.0.0', port=5000)

