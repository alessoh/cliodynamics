"""
Data collection module for historical and current indicators
In production, would connect to real data sources (FRED, World Bank, etc.)
For demo purposes, generates realistic synthetic data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and processes data for cliodynamics analysis"""
    
    def __init__(self, config):
        self.config = config
        self.data_cache = {}
        
    def collect_historical_data(self, start_year, end_year):
        """Collect historical data for analysis period"""
        logger.info(f"Collecting data from {start_year} to {end_year}")
        
        # Check cache first
        cache_key = f"{start_year}_{end_year}"
        if cache_key in self.data_cache:
            logger.info("Using cached data")
            return self.data_cache[cache_key]
        
        # Generate year range
        years = list(range(start_year, end_year + 1))
        
        # Collect different data types
        economic_data = self._collect_economic_data(years)
        political_data = self._collect_political_data(years)
        social_data = self._collect_social_data(years)
        demographic_data = self._collect_demographic_data(years)
        
        # Combine all data
        historical_data = {
            'years': years,
            'economic': economic_data,
            'political': political_data,
            'social': social_data,
            'demographic': demographic_data,
            'events': self._get_historical_events(start_year, end_year)
        }
        
        # Cache the data
        self.data_cache[cache_key] = historical_data
        
        # Save to file for persistence
        self._save_data(historical_data, start_year, end_year)
        
        return historical_data
    
    def _collect_economic_data(self, years):
        """Collect economic indicators"""
        logger.info("Collecting economic data...")
        
        data = {
            'gdp_growth': [],
            'inequality_gini': [],
            'unemployment': [],
            'real_wages': [],
            'debt_to_gdp': [],
            'inflation': []
        }
        
        for i, year in enumerate(years):
            # GDP growth (with business cycles)
            trend = 2.5
            cycle = 2 * np.sin(2 * np.pi * i / 10)  # 10-year cycle
            shock = 0
            if year in [1929, 1973, 1979, 2008, 2020]:
                shock = -5
            gdp_growth = trend + cycle + shock + np.random.normal(0, 1)
            data['gdp_growth'].append(gdp_growth)
            
            # Inequality (trending up)
            base_gini = 0.35
            trend = (year - 1800) * 0.0003
            cycle = 0.05 * np.sin(2 * np.pi * i / 30)
            gini = base_gini + trend + cycle + np.random.normal(0, 0.02)
            data['inequality_gini'].append(np.clip(gini, 0.2, 0.6))
            
            # Unemployment
            base = 5
            cycle = 3 * np.sin(2 * np.pi * i / 7)  # 7-year cycle
            if year in [1933, 1982, 2009, 2020]:
                base = 15
            unemployment = base + cycle + np.random.normal(0, 0.5)
            data['unemployment'].append(np.clip(unemployment, 2, 25))
            
            # Real wages (stagnating since 1970s)
            if year < 1970:
                wage_growth = 2.5 + np.random.normal(0, 1)
            else:
                wage_growth = 0.5 + np.random.normal(0, 1)
            data['real_wages'].append(wage_growth)
            
            # Debt to GDP (trending up)
            base_debt = 30
            if year > 1980:
                base_debt = 30 + (year - 1980) * 1.5
            debt = base_debt + np.random.normal(0, 5)
            data['debt_to_gdp'].append(np.clip(debt, 20, 150))
            
            # Inflation
            base_inflation = 2
            if 1970 <= year <= 1982:
                base_inflation = 8
            elif year in [2021, 2022, 2023]:
                base_inflation = 6
            inflation = base_inflation + np.random.normal(0, 1)
            data['inflation'].append(np.clip(inflation, -2, 15))
        
        return data
    
    def _collect_political_data(self, years):
        """Collect political indicators"""
        logger.info("Collecting political data...")
        
        data = {
            'political_violence_events': [],
            'polarization_index': [],
            'trust_in_government': [],
            'protest_frequency': [],
            'elite_turnover': []
        }
        
        for i, year in enumerate(years):
            # Political violence (50-year cycles)
            base = 5
            cycle = 10 * np.sin(2 * np.pi * (year - 1870) / 50)
            if year in range(1861, 1866) or year in range(1968, 1973):
                base = 50
            violence = base + cycle + np.random.poisson(2)
            data['political_violence_events'].append(int(np.clip(violence, 0, 100)))
            
            # Polarization (increasing trend)
            base_polar = 30
            if year > 1960:
                base_polar = 30 + (year - 1960) * 0.5
            polarization = base_polar + np.random.normal(0, 5)
            data['polarization_index'].append(np.clip(polarization, 0, 100))
            
            # Trust in government (declining trend)
            base_trust = 70
            if year > 1960:
                base_trust = 70 - (year - 1960) * 0.5
            trust = base_trust + np.random.normal(0, 5)
            data['trust_in_government'].append(np.clip(trust, 10, 90))
            
            # Protest frequency
            base_protest = 10
            if year in range(1960, 1975) or year in range(2010, 2025):
                base_protest = 30
            protests = base_protest + np.random.poisson(5)
            data['protest_frequency'].append(int(protests))
            
            # Elite turnover
            base_turnover = 15
            if year % 4 == 0:  # Election years
                base_turnover = 25
            turnover = base_turnover + np.random.normal(0, 5)
            data['elite_turnover'].append(np.clip(turnover, 5, 50))
        
        return data
    
    def _collect_social_data(self, years):
        """Collect social indicators"""
        logger.info("Collecting social data...")
        
        data = {
            'social_mobility': [],
            'education_inequality': [],
            'health_inequality': [],
            'cultural_conflict_index': [],
            'generational_wealth_gap': []
        }
        
        for i, year in enumerate(years):
            # Social mobility (declining)
            base_mobility = 60
            if year > 1970:
                base_mobility = 60 - (year - 1970) * 0.3
            mobility = base_mobility + np.random.normal(0, 5)
            data['social_mobility'].append(np.clip(mobility, 20, 80))
            
            # Education inequality (increasing)
            base_edu = 30
            if year > 1980:
                base_edu = 30 + (year - 1980) * 0.5
            edu_ineq = base_edu + np.random.normal(0, 5)
            data['education_inequality'].append(np.clip(edu_ineq, 20, 80))
            
            # Health inequality
            base_health = 25
            if year > 1970:
                base_health = 25 + (year - 1970) * 0.3
            health_ineq = base_health + np.random.normal(0, 5)
            data['health_inequality'].append(np.clip(health_ineq, 15, 70))
            
            # Cultural conflict
            base_culture = 30
            cycle = 15 * np.sin(2 * np.pi * i / 30)
            culture = base_culture + cycle + np.random.normal(0, 5)
            data['cultural_conflict_index'].append(np.clip(culture, 10, 80))
            
            # Generational wealth gap
            base_gap = 20
            if year > 1980:
                base_gap = 20 + (year - 1980) * 0.8
            gap = base_gap + np.random.normal(0, 5)
            data['generational_wealth_gap'].append(np.clip(gap, 10, 90))
        
        return data
    
    def _collect_demographic_data(self, years):
        """Collect demographic indicators"""
        logger.info("Collecting demographic data...")
        
        data = {
            'youth_bulge': [],
            'urbanization_rate': [],
            'immigration_rate': [],
            'dependency_ratio': [],
            'elite_overproduction_proxy': []
        }
        
        for i, year in enumerate(years):
            # Youth bulge (percentage of population 15-29)
            base_youth = 25
            if 1960 <= year <= 1980:
                base_youth = 30
            elif 2000 <= year <= 2020:
                base_youth = 22
            youth = base_youth + np.random.normal(0, 2)
            data['youth_bulge'].append(np.clip(youth, 15, 35))
            
            # Urbanization rate
            urban = 20 + (year - 1800) * 0.3 + np.random.normal(0, 2)
            data['urbanization_rate'].append(np.clip(urban, 10, 90))
            
            # Immigration rate
            base_immig = 5
            if 1880 <= year <= 1920:
                base_immig = 15
            elif 1990 <= year <= 2020:
                base_immig = 10
            immigration = base_immig + np.random.normal(0, 2)
            data['immigration_rate'].append(np.clip(immigration, 0, 25))
            
            # Dependency ratio
            base_dep = 50
            if year > 2000:
                base_dep = 50 + (year - 2000) * 0.5
            dependency = base_dep + np.random.normal(0, 5)
            data['dependency_ratio'].append(np.clip(dependency, 30, 80))
            
            # Elite overproduction proxy (college grads vs elite positions)
            base_elite = 20
            if year > 1960:
                base_elite = 20 + (year - 1960) * 0.6
            elite_proxy = base_elite + np.random.normal(0, 5)
            data['elite_overproduction_proxy'].append(np.clip(elite_proxy, 10, 90))
        
        return data
    
    def _get_historical_events(self, start_year, end_year):
        """Get major historical events in the time period"""
        all_events = [
            (1861, "American Civil War begins"),
            (1865, "American Civil War ends"),
            (1914, "World War I begins"),
            (1917, "Russian Revolution"),
            (1918, "World War I ends"),
            (1929, "Great Depression begins"),
            (1933, "New Deal begins"),
            (1939, "World War II begins"),
            (1945, "World War II ends"),
            (1968, "Global protests and unrest"),
            (1973, "Oil crisis"),
            (1989, "Fall of Berlin Wall"),
            (1991, "Soviet Union collapses"),
            (2001, "9/11 attacks"),
            (2008, "Global Financial Crisis"),
            (2011, "Arab Spring"),
            (2016, "Brexit vote, Trump election"),
            (2020, "COVID-19 pandemic begins"),
            (2021, "January 6 Capitol riot"),
            (2022, "Russia invades Ukraine"),
            (2023, "AI revolution accelerates")
        ]
        
        # Filter events within the time period
        events = []
        for year, event in all_events:
            if start_year <= year <= end_year:
                events.append({
                    'year': year,
                    'event': event,
                    'impact_score': self._calculate_event_impact(event)
                })
        
        return events
    
    def _calculate_event_impact(self, event):
        """Calculate impact score for historical events"""
        high_impact_keywords = ['war', 'revolution', 'crisis', 'depression', 'pandemic']
        medium_impact_keywords = ['election', 'protest', 'riot', 'collapse']
        
        event_lower = event.lower()
        
        for keyword in high_impact_keywords:
            if keyword in event_lower:
                return np.random.randint(70, 100)
        
        for keyword in medium_impact_keywords:
            if keyword in event_lower:
                return np.random.randint(40, 70)
        
        return np.random.randint(20, 40)
    
    def _save_data(self, data, start_year, end_year):
        """Save collected data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/data/historical_data_{start_year}_{end_year}_{timestamp}.json"
        
        # Ensure directory exists
        Path("output/data").mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(data)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Data saved to {filename}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_real_time_indicators(self):
        """Get current real-time indicators (simulated)"""
        logger.info("Fetching real-time indicators...")
        
        # In production, would fetch from APIs
        current_indicators = {
            'timestamp': datetime.now().isoformat(),
            'market_volatility': np.random.uniform(15, 35),
            'social_media_sentiment': np.random.uniform(-0.5, 0.5),
            'news_conflict_score': np.random.uniform(20, 60),
            'economic_policy_uncertainty': np.random.uniform(50, 150),
            'google_trends_unrest': np.random.uniform(10, 50)
        }
        
        return current_indicators
    
    def load_cached_data(self, filename):
        """Load previously collected data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded cached data from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}")
            return None