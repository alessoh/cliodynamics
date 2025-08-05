"""
Utility functions for the Cliodynamics Analysis System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from colorama import init, Fore, Style
import pandas as pd
from tabulate import tabulate

# Initialize colorama for Windows
init()

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/cliodynamics_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add colors to console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    
    # Get root logger
    logger = logging.getLogger()
    logger.handlers[1] = console_handler  # Replace stdout handler
    
    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    FORMATS = {
        logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
        logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def create_output_dirs():
    """Create necessary output directories"""
    directories = [
        'output',
        'output/data',
        'output/visualizations',
        'output/reports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("Output directories created")

def format_analysis_report(analysis_results):
    """Format analysis results for display"""
    report = []
    
    # Header
    report.append("=" * 80)
    report.append("CLIODYNAMICS ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Metadata
    metadata = analysis_results.get('metadata', {})
    report.append(f"Analysis Date: {metadata.get('analysis_date', 'N/A')}")
    report.append(f"Period: {metadata.get('start_year', 'N/A')} - {metadata.get('end_year', 'N/A')}")
    report.append(f"Scenario: {metadata.get('scenario', 'N/A').capitalize()}")
    report.append("")
    
    # Current State
    report.append("CURRENT STATE ASSESSMENT")
    report.append("-" * 40)
    
    current = analysis_results.get('current_state', {})
    state_data = [
        ['Indicator', 'Value', 'Status'],
        ['Political Stress Index', f"{current.get('psi', 0):.1f}%", get_status_label(current.get('psi', 0))],
        ['Elite Overproduction', f"{current.get('elite_overproduction', 0):.1f}%", 
         get_status_label(current.get('elite_overproduction', 0))],
        ['Popular Immiseration', f"{current.get('popular_immiseration', 0):.1f}%", 
         get_status_label(current.get('popular_immiseration', 0))],
        ['State Fiscal Health', f"{current.get('state_fiscal_health', 0):.1f}%", 
         get_status_label(100 - current.get('state_fiscal_health', 0))],
        ['Intra-Elite Competition', f"{current.get('intra_elite_competition', 0):.1f}%", 
         get_status_label(current.get('intra_elite_competition', 0))]
    ]
    
    report.append(tabulate(state_data, headers='firstrow', tablefmt='grid'))
    report.append("")
    
    # Cycles
    report.append("CYCLE ANALYSIS")
    report.append("-" * 40)
    
    cycles = analysis_results.get('cycles', {})
    report.append(f"Secular Cycle Phase: {cycles.get('secular_phase', 'Unknown')}")
    report.append(f"Position in Cycle: {cycles.get('secular_position', 0):.1f}%")
    report.append(f"Years to Violence Peak: {cycles.get('years_to_violence_peak', 'N/A')}")
    report.append(f"Trend Direction: {cycles.get('trend_direction', 'Unknown').capitalize()}")
    report.append("")
    
    # AI Insights
    report.append("AI ANALYSIS INSIGHTS")
    report.append("-" * 40)
    
    insights = analysis_results.get('llm_insights', {})
    report.append(insights.get('summary', 'No summary available'))
    report.append("")
    
    if insights.get('risk_factors'):
        report.append("Key Risk Factors:")
        for i, risk in enumerate(insights['risk_factors'][:5], 1):
            report.append(f"  {i}. {risk}")
        report.append("")
    
    # Recommendations
    if insights.get('recommendations'):
        report.append("TOP RECOMMENDATIONS")
        report.append("-" * 40)
        
        for rec in insights['recommendations'][:3]:
            report.append(f"\n{rec.get('area', 'Unknown')} (Priority: {rec.get('priority', 'Unknown')})")
            report.append(f"  {rec.get('recommendation', '')}")
            
            if rec.get('specific_actions'):
                report.append("  Actions:")
                for action in rec['specific_actions'][:3]:
                    report.append(f"    • {action}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def get_status_label(value):
    """Get status label for indicator value"""
    if value < 20:
        return f"{Fore.GREEN}Very Low{Style.RESET_ALL}"
    elif value < 40:
        return f"{Fore.GREEN}Low{Style.RESET_ALL}"
    elif value < 60:
        return f"{Fore.YELLOW}Moderate{Style.RESET_ALL}"
    elif value < 80:
        return f"{Fore.RED}High{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{Style.BRIGHT}CRITICAL{Style.RESET_ALL}"

def calculate_trend(data, window=10):
    """Calculate trend direction from time series data"""
    if len(data) < window:
        return 0
    
    recent = np.mean(data[-window//2:])
    older = np.mean(data[-window:-window//2])
    
    return (recent - older) / older * 100 if older != 0 else 0

def interpolate_missing_data(data, method='linear'):
    """Interpolate missing data points"""
    df = pd.DataFrame(data)
    df = df.interpolate(method=method, limit_direction='both')
    return df.to_dict('list')

def validate_api_keys():
    """Validate that required API keys are present"""
    from config import Config
    
    try:
        config = Config()
        
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found")
        
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        logging.info("API keys validated successfully")
        return True
        
    except Exception as e:
        logging.error(f"API key validation failed: {str(e)}")
        print(f"\n{Fore.RED}ERROR: {str(e)}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Please create a .env file with your API keys:{Style.RESET_ALL}")
        print("OPENAI_API_KEY=your-openai-key")
        print("ANTHROPIC_API_KEY=your-anthropic-key\n")
        return False

def save_checkpoint(data, checkpoint_name):
    """Save analysis checkpoint for recovery"""
    checkpoint_dir = Path('output/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = checkpoint_dir / f"{checkpoint_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logging.info(f"Checkpoint saved: {filename}")
    return filename

def load_checkpoint(checkpoint_path):
    """Load analysis checkpoint"""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Checkpoint loaded: {checkpoint_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {str(e)}")
        return None

def generate_summary_statistics(indicators):
    """Generate summary statistics for indicators"""
    stats = {}
    
    for key in ['psi', 'elite_overproduction', 'popular_immiseration', 
                'state_fiscal_health', 'intra_elite_competition']:
        if key in indicators:
            data = indicators[key]
            stats[key] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'current': data[-1],
                'trend': calculate_trend(data)
            }
    
    return stats

def export_to_csv(data, filename):
    """Export data to CSV format"""
    df = pd.DataFrame(data)
    
    output_path = Path('output/data') / f"{filename}.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Data exported to {output_path}")
    return output_path

def compare_scenarios(scenarios_data):
    """Compare multiple scenario projections"""
    comparison = {}
    
    for scenario_name, scenario_data in scenarios_data.items():
        final_psi = scenario_data['psi'][-1]
        max_psi = max(scenario_data['psi'])
        avg_psi = np.mean(scenario_data['psi'])
        
        comparison[scenario_name] = {
            'final_psi': final_psi,
            'max_psi': max_psi,
            'avg_psi': avg_psi,
            'crisis_years': sum(1 for p in scenario_data['psi'] if p > 80)
        }
    
    return comparison

def format_time_remaining(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds/60)} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def check_system_requirements():
    """Check if system meets requirements"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 12):
        issues.append(f"Python 3.12+ required (found {sys.version})")
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'plotly', 'dash',
        'openai', 'anthropic', 'scipy', 'scikit-learn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check output directories
    if not Path('output').exists():
        create_output_dirs()
    
    if issues:
        print(f"\n{Fore.YELLOW}System requirement issues found:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n{Fore.CYAN}Run 'pip install -r requirements.txt' to install missing packages{Style.RESET_ALL}\n")
        return False
    
    return True

def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}{'='*60}
{Fore.BLUE}     CLIODYNAMICS ANALYSIS SYSTEM     
{Fore.CYAN}{'='*60}{Style.RESET_ALL}
{Fore.GREEN}Analyzing Historical Patterns & Social Cycles{Style.RESET_ALL}
{Fore.YELLOW}Based on Structural-Demographic Theory{Style.RESET_ALL}
{Fore.CYAN}{'='*60}{Style.RESET_ALL}
"""
    print(banner)

if __name__ == "__main__":
    # Test utilities
    print_banner()
    
    if check_system_requirements():
        print(f"{Fore.GREEN}✓ All system requirements met{Style.RESET_ALL}")
    
    if validate_api_keys():
        print(f"{Fore.GREEN}✓ API keys validated{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Utilities module ready{Style.RESET_ALL}")