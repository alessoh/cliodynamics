"""
Core cliodynamics calculations and models
Based on structural-demographic theory
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CliodynamicsEngine:
    """Core engine for cliodynamics calculations"""
    
    def __init__(self, config):
        self.config = config
        self.current_year = config.CURRENT_YEAR
        
    def calculate_indicators(self, historical_data):
        """Calculate all structural-demographic indicators"""
        logger.info("Calculating structural-demographic indicators...")
        
        years = historical_data['years']
        
        # Initialize indicators
        indicators = {
            'years': years,
            'psi': [],
            'elite_overproduction': [],
            'popular_immiseration': [],
            'state_fiscal_health': [],
            'intra_elite_competition': []
        }
        
        # Calculate each indicator
        for i, year in enumerate(years):
            # Elite Overproduction
            elite_op = self._calculate_elite_overproduction(year, historical_data, i)
            indicators['elite_overproduction'].append(elite_op)
            
            # Popular Immiseration
            immiseration = self._calculate_immiseration(year, historical_data, i)
            indicators['popular_immiseration'].append(immiseration)
            
            # State Fiscal Health
            fiscal = self._calculate_fiscal_health(year, historical_data, i)
            indicators['state_fiscal_health'].append(fiscal)
            
            # Intra-Elite Competition (pass the indicators being built)
            competition = self._calculate_elite_competition(year, indicators, i)
            indicators['intra_elite_competition'].append(competition)
            
            # Political Stress Indicator (weighted combination)
            psi = self._calculate_psi(elite_op, immiseration, fiscal, competition)
            indicators['psi'].append(psi)
        
        # Smooth indicators
        for key in ['psi', 'elite_overproduction', 'popular_immiseration', 
                    'state_fiscal_health', 'intra_elite_competition']:
            indicators[key] = self._smooth_series(indicators[key])
            
        return indicators
    
    def _calculate_elite_overproduction(self, year, data, index):
        """Calculate elite overproduction index"""
        # Base trend (increasing over time)
        base = 30 + (year - 1800) * 0.15
        
        # Secular cycle component
        secular_phase = ((year - 1800) % self.config.SECULAR_CYCLE_LENGTH) / self.config.SECULAR_CYCLE_LENGTH
        secular_component = 20 * np.sin(secular_phase * 2 * np.pi)
        
        # Add historical shocks
        shock = 0
        for start, end, event in self.config.MAJOR_CRISIS_YEARS:
            if start <= year <= end:
                shock += 15
        
        # Random variation
        noise = np.random.normal(0, 5)
        
        value = base + secular_component + shock + noise
        return np.clip(value, 0, 100)
    
    def _calculate_immiseration(self, year, data, index):
        """Calculate popular immiseration index"""
        # Base trend
        base = 40 + (year - 1900) * 0.05
        
        # Economic cycles
        economic_cycle = 15 * np.sin(((year - 1800) / 20) * 2 * np.pi)
        
        # Inverse relationship with fiscal health
        fiscal_effect = -0.3 * (year - 1950) * 0.1 if year > 1950 else 0
        
        # Historical events
        shock = 0
        for start, end, event in self.config.MAJOR_CRISIS_YEARS:
            if start <= year <= end:
                shock += 20
        
        noise = np.random.normal(0, 5)
        
        value = base + economic_cycle + fiscal_effect + shock + noise
        return np.clip(value, 0, 100)
    
    def _calculate_fiscal_health(self, year, data, index):
        """Calculate state fiscal health"""
        # Base trend (declining over time due to increasing obligations)
        base = 70 - (year - 1900) * 0.15
        
        # War and crisis effects
        crisis_effect = 0
        for start, end, event in self.config.MAJOR_CRISIS_YEARS:
            if start <= year <= end + 5:  # Fiscal effects last longer
                crisis_effect -= 15
        
        # Economic boom periods
        if 1920 <= year <= 1929 or 1945 <= year <= 1970 or 1990 <= year <= 2000:
            crisis_effect += 10
        
        noise = np.random.normal(0, 5)
        
        value = base + crisis_effect + noise
        return np.clip(value, 0, 100)
    
    def _calculate_elite_competition(self, year, data, index):
        """Calculate intra-elite competition"""
        # Correlated with elite overproduction but with lag
        if index > 10 and 'elite_overproduction' in data:
            elite_op_data = data.get('elite_overproduction', [])
            if len(elite_op_data) > index - 10:
                elite_op_avg = np.mean(elite_op_data[max(0, index-10):index])
            else:
                elite_op_avg = 50
        else:
            elite_op_avg = 50
        
        base = elite_op_avg * 0.8
        
        # Political cycle effects
        political_cycle = 10 * np.sin(((year - 1800) / 4) * 2 * np.pi)  # 4-year cycles
        
        noise = np.random.normal(0, 5)
        
        value = base + political_cycle + noise
        return np.clip(value, 0, 100)
    
    def _calculate_psi(self, elite_op, immiseration, fiscal, competition):
        """Calculate Political Stress Indicator"""
        weights = self.config.PSI_WEIGHTS
        
        # Invert fiscal health (low fiscal health = high stress)
        fiscal_stress = 100 - fiscal
        
        psi = (weights['elite_overproduction'] * elite_op +
               weights['popular_immiseration'] * immiseration +
               weights['state_fiscal_health'] * fiscal_stress +
               weights['intra_elite_competition'] * competition)
        
        return np.clip(psi, 0, 100)
    
    def _smooth_series(self, data, window=5):
        """Apply smoothing to time series"""
        if len(data) < window:
            return data
        
        # Convert to numpy array and check for NaN values
        data_array = np.array(data)
        
        # Replace NaN values with interpolation
        if np.any(np.isnan(data_array)):
            nans = np.isnan(data_array)
            if not np.all(nans):
                data_array[nans] = np.interp(np.where(nans)[0], 
                                             np.where(~nans)[0], 
                                             data_array[~nans])
            else:
                # If all NaN, replace with default value
                data_array[:] = 50
        
        # Use Savitzky-Golay filter for smoothing while preserving features
        try:
            smoothed = signal.savgol_filter(data_array, window, min(3, window-1))
            return smoothed.tolist()
        except:
            # If smoothing fails, return original data
            return data
    
    def identify_cycles(self, indicators):
        """Identify secular and violence cycles"""
        logger.info("Identifying historical cycles...")
        
        years = indicators['years']
        psi = indicators['psi']
        current_year_idx = years.index(self.current_year) if self.current_year in years else -1
        
        # Secular cycle analysis
        secular_position = ((self.current_year - 1800) % self.config.SECULAR_CYCLE_LENGTH) / self.config.SECULAR_CYCLE_LENGTH
        
        if secular_position < 0.25:
            phase = "Expansion"
        elif secular_position < 0.5:
            phase = "Stagflation"
        elif secular_position < 0.75:
            phase = "Crisis"
        else:
            phase = "Depression"
        
        # Violence cycle analysis
        violence_position = ((self.current_year - 1870) % self.config.VIOLENCE_CYCLE_LENGTH) / self.config.VIOLENCE_CYCLE_LENGTH
        years_to_peak = int((1.0 - violence_position) * self.config.VIOLENCE_CYCLE_LENGTH)
        if years_to_peak == 0:
            years_to_peak = self.config.VIOLENCE_CYCLE_LENGTH
        
        # Find historical cycle peaks
        peaks, _ = signal.find_peaks(psi, distance=20, prominence=10)
        peak_years = [years[i] for i in peaks]
        
        # Analyze cycle characteristics
        cycles = {
            'secular_phase': phase,
            'secular_position': secular_position * 100,
            'violence_position': violence_position * 100,
            'years_to_violence_peak': years_to_peak,
            'historical_peaks': peak_years,
            'current_psi': psi[current_year_idx] if current_year_idx >= 0 else psi[-1],
            'cycle_amplitude': np.std(psi),
            'trend_direction': 'increasing' if np.mean(psi[-10:]) > np.mean(psi[-20:-10]) else 'decreasing'
        }
        
        return cycles
    
    def generate_projections(self, indicators, cycles, scenario='baseline'):
        """Generate future projections based on scenario"""
        logger.info(f"Generating {scenario} projections...")
        
        years = indicators['years']
        last_year = years[-1]
        projection_years = list(range(last_year + 1, last_year + 31))
        
        # Get scenario parameters
        trends = self.config.PROJECTION_SCENARIOS[scenario]
        
        projections = {
            'years': projection_years,
            'psi': [],
            'elite_overproduction': [],
            'popular_immiseration': [],
            'state_fiscal_health': [],
            'intra_elite_competition': [],
            'confidence_intervals': {
                'psi_lower': [],
                'psi_upper': []
            }
        }
        
        # Get last values
        last_values = {
            'elite_overproduction': indicators['elite_overproduction'][-1],
            'popular_immiseration': indicators['popular_immiseration'][-1],
            'state_fiscal_health': indicators['state_fiscal_health'][-1],
            'intra_elite_competition': indicators['intra_elite_competition'][-1]
        }
        
        # Project each year
        for i, year in enumerate(projection_years):
            # Update each indicator based on trends
            elite_op = last_values['elite_overproduction'] + trends['elite_overproduction_trend'] * (i + 1)
            elite_op += np.random.normal(0, 2)  # Add uncertainty
            elite_op = np.clip(elite_op, 0, 100)
            projections['elite_overproduction'].append(elite_op)
            
            immiseration = last_values['popular_immiseration'] + trends['immiseration_trend'] * (i + 1)
            immiseration += np.random.normal(0, 2)
            immiseration = np.clip(immiseration, 0, 100)
            projections['popular_immiseration'].append(immiseration)
            
            fiscal = last_values['state_fiscal_health'] + trends['fiscal_health_trend'] * (i + 1)
            fiscal += np.random.normal(0, 2)
            fiscal = np.clip(fiscal, 0, 100)
            projections['state_fiscal_health'].append(fiscal)
            
            competition = last_values['intra_elite_competition'] + trends['competition_trend'] * (i + 1)
            competition += np.random.normal(0, 2)
            competition = np.clip(competition, 0, 100)
            projections['intra_elite_competition'].append(competition)
            
            # Calculate PSI
            psi = self._calculate_psi(elite_op, immiseration, fiscal, competition)
            projections['psi'].append(psi)
            
            # Calculate confidence intervals (wider as we go further)
            uncertainty = 5 + i * 0.5
            projections['confidence_intervals']['psi_lower'].append(max(0, psi - uncertainty))
            projections['confidence_intervals']['psi_upper'].append(min(100, psi + uncertainty))
        
        # Add cycle effects
        projections = self._add_cycle_effects(projections, cycles)
        
        return projections
    
    def _add_cycle_effects(self, projections, cycles):
        """Add secular and violence cycle effects to projections"""
        years = projections['years']
        
        for i, year in enumerate(years):
            # Secular cycle effect
            secular_pos = ((year - 1800) % self.config.SECULAR_CYCLE_LENGTH) / self.config.SECULAR_CYCLE_LENGTH
            secular_effect = 10 * np.sin(secular_pos * 2 * np.pi)
            
            # Violence cycle effect
            violence_pos = ((year - 1870) % self.config.VIOLENCE_CYCLE_LENGTH) / self.config.VIOLENCE_CYCLE_LENGTH
            violence_effect = 5 * np.sin(violence_pos * 2 * np.pi)
            
            # Apply effects
            projections['psi'][i] += secular_effect + violence_effect
            projections['psi'][i] = np.clip(projections['psi'][i], 0, 100)
            
            # Update confidence intervals
            projections['confidence_intervals']['psi_lower'][i] = np.clip(
                projections['confidence_intervals']['psi_lower'][i] + secular_effect + violence_effect, 0, 100)
            projections['confidence_intervals']['psi_upper'][i] = np.clip(
                projections['confidence_intervals']['psi_upper'][i] + secular_effect + violence_effect, 0, 100)
        
        return projections