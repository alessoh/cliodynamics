"""
Configuration settings for the AI Cliodynamics Analysis System
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Also try loading from current directory
load_dotenv()

class Config:
    """Configuration class for the cliodynamics system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Debug output
    if ANTHROPIC_API_KEY:
        print(f"Config: Anthropic key loaded, length: {len(ANTHROPIC_API_KEY)}, starts with: {ANTHROPIC_API_KEY[:10]}...")
    else:
        print("Config: No Anthropic key found in environment")
    
    # Validate API keys
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    
    # Model Selection
    OPENAI_MODEL = "gpt-4"  # or "gpt-4-turbo", "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"  # Using a model that definitely exists
    
    # Analysis Parameters
    DEFAULT_START_YEAR = 1800
    DEFAULT_END_YEAR = datetime.now().year + 30
    CURRENT_YEAR = datetime.now().year
    
    # Cliodynamics Model Parameters
    SECULAR_CYCLE_LENGTH = 250  # years
    VIOLENCE_CYCLE_LENGTH = 50  # years
    
    # Indicator Thresholds
    PSI_CRITICAL = 80
    PSI_HIGH = 60
    PSI_MEDIUM = 40
    PSI_LOW = 20
    
    # Weights for PSI calculation
    PSI_WEIGHTS = {
        'elite_overproduction': 0.3,
        'popular_immiseration': 0.3,
        'state_fiscal_health': 0.2,
        'intra_elite_competition': 0.2
    }
    
    # Historical Events (for calibration)
    MAJOR_CRISIS_YEARS = [
        (1861, 1865, "American Civil War"),
        (1917, 1921, "Russian Revolution & Civil War"),
        (1929, 1939, "Great Depression"),
        (1968, 1972, "Late 1960s Unrest"),
        (2008, 2012, "Great Recession"),
        (2020, 2023, "COVID & Social Unrest")
    ]
    
    # Projection Parameters
    PROJECTION_SCENARIOS = {
        'optimistic': {
            'elite_overproduction_trend': -0.5,
            'immiseration_trend': -0.3,
            'fiscal_health_trend': 0.4,
            'competition_trend': -0.4
        },
        'baseline': {
            'elite_overproduction_trend': 0.2,
            'immiseration_trend': 0.1,
            'fiscal_health_trend': -0.1,
            'competition_trend': 0.2
        },
        'pessimistic': {
            'elite_overproduction_trend': 0.5,
            'immiseration_trend': 0.4,
            'fiscal_health_trend': -0.3,
            'competition_trend': 0.5
        }
    }
    
    # Chart Settings (reduced DPI to prevent oversized images)
    CHART_STYLE = 'dark_background'
    CHART_DPI = 150  # Reduced from 300
    CHART_SIZE = (12, 8)
    
    # Dashboard Settings
    DASHBOARD_HOST = '127.0.0.1'
    DASHBOARD_PORT = 8050
    DASHBOARD_DEBUG = True
    
    # Output Settings
    OUTPUT_DIR = 'output'
    VISUALIZATION_DIR = 'output/visualizations'
    REPORT_DIR = 'output/reports'
    DATA_DIR = 'output/data'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # LLM Prompt Templates
    HISTORICAL_ANALYSIS_PROMPT = """
    Analyze the following cliodynamics indicators and historical patterns:
    
    Time Period: {start_year} - {end_year}
    Current PSI: {psi}%
    Elite Overproduction: {elite}%
    Popular Immiseration: {immiseration}%
    State Fiscal Health: {fiscal}%
    Current Secular Phase: {phase}
    
    Historical Context:
    {historical_events}
    
    Please provide:
    1. Historical parallels to current conditions
    2. Key drivers of current stress levels
    3. Potential tipping points
    4. Recommended interventions
    5. Likely scenarios for next 10-30 years
    
    Use structural-demographic theory principles in your analysis.
    """
    
    PROJECTION_ANALYSIS_PROMPT = """
    Based on these cliodynamics projections for the {scenario} scenario:
    
    {projection_data}
    
    Analyze:
    1. Likelihood of this scenario
    2. Key assumptions and vulnerabilities
    3. Potential black swan events
    4. Policy interventions that could alter trajectory
    5. Historical precedents for similar trajectories
    """
    
    # Data Sources (placeholder - would connect to real APIs)
    DATA_SOURCES = {
        'economic': 'simulated',  # Would use FRED, World Bank, etc.
        'political': 'simulated',  # Would use GDELT, news APIs, etc.
        'social': 'simulated'      # Would use social media APIs, surveys, etc.
    }