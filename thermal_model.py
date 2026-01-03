
import numpy as np


class ThermalModel:
    """
    Thermal model for calculating heat sink performance and junction temperature.
    """
    
    def __init__(self):
        """Initialize with default values from the reference."""
        # Processor Die Dimensions
        self.die_length = 52.5e-3  # m
        self.die_width = 45e-3  # m
        self.die_thickness = 2.2e-3  # m
        self.die_area = self.die_length * self.die_width  # m²
        
        # Heat Sink Specifications
        self.sink_length = 90e-3  # m
        self.sink_width = 116e-3  # m
        self.base_thickness = 2.5e-3  # m
        self.base_area = self.sink_length * self.sink_width  # m²
        self.num_fins = 60
        self.fin_thickness = 0.8e-3  # m
        self.overall_height = 27e-3  # m
        self.fin_height = 24.5e-3  # m
        
        # Calculate fin spacing
        self.fin_spacing = (self.sink_width - (self.num_fins * self.fin_thickness)) / (self.num_fins - 1)
        
        # Thermal Design Power
        self.tdp = 150  # W
        
        # Material Properties
        self.k_al = 167  # W/m·K (Aluminum 6061-T6)
        self.k_tim = 4  # W/m·K (Thermal grease)
        self.tim_thickness = 0.1e-3  # m
        
        # Air Properties at 25°C
        self.t_ambient = 25  # °C
        self.k_air = 0.0262  # W/m·K
        self.kinematic_viscosity = 1.57e-5  # m²/s
        self.prandtl_number = 0.71
        self.air_velocity = 1  # m/s
        
        # Junction-to-case resistance (from Excel: 0.2 °C/W, PDF mentions 0.1)
        self.r_jc = 0.2  # °C/W
    
    def calculate_r_tim(self):
        """
        Calculate thermal resistance of Thermal Interface Material (TIM).
        
        Returns:
            float: TIM resistance in °C/W
        """
        r_tim = self.tim_thickness / (self.k_tim * self.die_area)
        return r_tim
    
    def calculate_r_cond(self):
        """
        Calculate conduction resistance through heat sink base.
        
        Returns:
            float: Conduction resistance in °C/W
        """
        r_cond = self.base_thickness / (self.k_al * self.die_area)
        return r_cond
    
    def calculate_reynolds_number(self):
        """
        Calculate Reynolds number for flow between fins.
        
        Returns:
            float: Reynolds number
        """
        re = (self.air_velocity * self.fin_spacing) / self.kinematic_viscosity
        return re
    
    def calculate_nusselt_number(self, re):
        """
        Calculate Nusselt number based on flow regime.
        
        Args:
            re: Reynolds number
            
        Returns:
            float: Nusselt number
        """
        if re < 2300:
            # Laminar flow - Sieder-Tate correlation
            # Using fin spacing as characteristic length for developing flow
            nu = 1.86 * (re * self.prandtl_number * (2 * self.fin_spacing / self.sink_length)) ** (1/3)
        else:
            # Turbulent flow - Dittus-Boelter equation
            nu = 0.023 * (re ** 0.8) * (self.prandtl_number ** 0.3)
        
        return nu
    
    def calculate_convective_heat_transfer_coefficient(self, nu):
        """
        Calculate convective heat transfer coefficient.
        
        Args:
            nu: Nusselt number
            
        Returns:
            float: Heat transfer coefficient in W/m²·K
        """
        # For channel flow, use 2*fin_spacing as characteristic length
        h = (nu * self.k_air) / (2 * self.fin_spacing)
        return h
    
    def calculate_convection_area(self):
        """
        Calculate total area available for convection.
        
        Returns:
            float: Total convection area in m²
        """
        # Area of single fin:
        # - Both sides: 2 * fin_height * sink_length
        # - Top edge: fin_thickness * sink_length
        area_single_fin = (2 * self.fin_height * self.sink_length) + (self.fin_thickness * self.sink_length)
        
        # Total fin area
        total_fin_area = self.num_fins * area_single_fin
        
        # Base area exposed to air (between fins)
        # Excel shows: Area of base = 0.00612 m²
        # Calculation: (sink_width - num_fins * fin_thickness) * sink_length
        base_exposed_area = (self.sink_width - self.num_fins * self.fin_thickness) * self.sink_length
        
        # Total convection area
        a_total = total_fin_area + base_exposed_area
        
        return a_total
    
    def calculate_r_conv(self):
        """
        Calculate convection resistance.
        
        Returns:
            float: Convection resistance in °C/W
        """
        re = self.calculate_reynolds_number()
        nu = self.calculate_nusselt_number(re)
        h = self.calculate_convective_heat_transfer_coefficient(nu)
        a_total = self.calculate_convection_area()
        
        r_conv = 1 / (h * a_total)
        return r_conv, re, nu, h, a_total
    
    def calculate_r_hs(self):
        """
        Calculate total heat sink thermal resistance.
        
        Returns:
            tuple: (R_hs, R_cond, R_conv, details)
        """
        r_cond = self.calculate_r_cond()
        r_conv, re, nu, h, a_total = self.calculate_r_conv()
        r_hs = r_cond + r_conv
        
        details = {
            'r_cond': r_cond,
            'r_conv': r_conv,
            'reynolds': re,
            'nusselt': nu,
            'h_coefficient': h,
            'convection_area': a_total
        }
        
        return r_hs, r_cond, r_conv, details
    
    def calculate_r_total(self):
        """
        Calculate total thermal resistance from junction to ambient.
        
        Returns:
            dict: Complete thermal resistance breakdown and results
        """
        r_tim = self.calculate_r_tim()
        r_hs, r_cond, r_conv, details = self.calculate_r_hs()
        
        r_total = self.r_jc + r_tim + r_hs
        
        results = {
            'r_jc': self.r_jc,
            'r_tim': r_tim,
            'r_cond': r_cond,
            'r_conv': r_conv,
            'r_hs': r_hs,
            'r_total': r_total,
            **details
        }
        
        return results
    
    def calculate_junction_temperature(self):
        """
        Calculate junction temperature.
        
        Returns:
            dict: Complete thermal analysis results
        """
        results = self.calculate_r_total()
        t_junction = self.t_ambient + (self.tdp * results['r_total'])
        results['t_junction'] = t_junction
        results['t_ambient'] = self.t_ambient
        results['tdp'] = self.tdp
        
        return results
    
    def print_results(self):
        """Print formatted results for validation."""
        results = self.calculate_junction_temperature()
        
        print("=" * 60)
        print("THERMAL MODEL RESULTS")
        print("=" * 60)
        print(f"\nThermal Resistances:")
        print(f"  R_jc (Junction-to-Case):     {results['r_jc']:.6f} °C/W")
        print(f"  R_TIM:                       {results['r_tim']:.6f} °C/W")
        print(f"  R_cond (Conduction):          {results['r_cond']:.6f} °C/W")
        print(f"  R_conv (Convection):         {results['r_conv']:.6f} °C/W")
        print(f"  R_hs (Heat Sink Total):      {results['r_hs']:.6f} °C/W")
        print(f"  R_total:                     {results['r_total']:.6f} °C/W")
        
        print(f"\nConvection Details:")
        print(f"  Reynolds Number:             {results['reynolds']:.6f}")
        print(f"  Nusselt Number:              {results['nusselt']:.6f}")
        print(f"  Heat Transfer Coefficient:   {results['h_coefficient']:.6f} W/m²·K")
        print(f"  Convection Area:             {results['convection_area']:.6f} m²")
        
        print(f"\nTemperature Results:")
        print(f"  Ambient Temperature:         {results['t_ambient']:.2f} °C")
        print(f"  Junction Temperature:        {results['t_junction']:.6f} °C")
        print(f"  Thermal Design Power:        {results['tdp']:.2f} W")
        
        print("\n" + "=" * 60)
        print("VALIDATION (Expected from Excel):")
        print("  Total Heat Sink Resistance:  0.373043 °C/W")
        print("  Junction Temperature:        80.956522 °C")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    # Create model instance
    model = ThermalModel()
    
    # Calculate and print results
    results = model.print_results()
    
    # Validation
    expected_r_total = 0.373043
    expected_t_junction = 80.956522
    
    print(f"\nValidation:")
    print(f"  R_total: Expected {expected_r_total:.6f}, Got {results['r_total']:.6f}, "
          f"Difference: {abs(results['r_total'] - expected_r_total):.6f} °C/W")
    print(f"  T_junction: Expected {expected_t_junction:.6f}, Got {results['t_junction']:.6f}, "
          f"Difference: {abs(results['t_junction'] - expected_t_junction):.6f} °C")

