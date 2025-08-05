# cliodynamics
# AI Cliodynamics Analysis System

A Python-based system that uses Large Language Models (OpenAI GPT and Claude) to perform cliodynamics analysis - the mathematical modeling of historical dynamics and social cycles based on structural-demographic theory.

## Overview

This project analyzes historical patterns and predicts future social dynamics by examining:
- Political Stress Indicators (PSI)
- Elite Overproduction
- Popular Immiseration  
- State Fiscal Health
- Intra-Elite Competition
- Secular Cycles (200-300 year patterns)
- 50-Year Violence Cycles

## Requirements

- Windows laptop with Python 3.12
- OpenAI API key
- Anthropic (Claude) API key
- Internet connection for API calls

## Installation

1. **Clone or download this repository**

2. **Install Python 3.12** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

3. **Open Command Prompt or PowerShell in the project directory**

4. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

5. **Install required packages**:
   ```
   pip install -r requirements.txt
   ```

6. **Set up your API keys**:
   - Create a `.env` file in the project root directory
   - Add your API keys:
     ```
     OPENAI_API_KEY=your-openai-api-key-here
     ANTHROPIC_API_KEY=your-anthropic-api-key-here
     ```
   - Replace the placeholder text with your actual API keys

## Quick Start

1. **Activate virtual environment** (if using):
   ```
   venv\Scripts\activate
   ```

2. **Run the main analysis**:
   ```
   python main.py
   ```

3. **View the interactive dashboard**:
   ```
   python dashboard.py
   ```
   - Opens at http://localhost:8050

## Project Structure

- `main.py` - Main entry point for running analyses
- `cliodynamics_core.py` - Core cliodynamics calculations and models
- `llm_analyzer.py` - LLM integration for historical analysis
- `data_collector.py` - Historical data collection and processing
- `dashboard.py` - Interactive web dashboard (Plotly Dash)
- `visualizer.py` - Chart generation and visualization
- `config.py` - Configuration settings
- `utils.py` - Utility functions
- `requirements.txt` - Python dependencies
- `.env` - API keys (create this file)
- `README.md` - This file

## Usage Examples

### Basic Analysis
```python
python main.py
```
This runs a complete cliodynamics analysis for the current time period.

### Custom Time Period Analysis
```python
python main.py --start-year 1900 --end-year 2050
```

### Generate Report
```python
python main.py --report
```
Creates a detailed PDF report with visualizations.

### Interactive Dashboard
```python
python dashboard.py
```
Launch the web dashboard for real-time exploration.

## Features

### 1. Historical Pattern Analysis
- Analyzes historical data from 1800 to present
- Identifies secular cycles and violence patterns
- Uses LLMs to interpret historical events

### 2. Current State Assessment
- Calculates current Political Stress Indicator (PSI)
- Evaluates elite overproduction levels
- Measures popular immiseration
- Assesses state fiscal health

### 3. Future Projections
- Projects trends 30 years into the future
- Provides optimistic, baseline, and pessimistic scenarios
- Identifies critical thresholds and tipping points

### 4. AI-Enhanced Analysis
- Uses GPT-4 and Claude for historical context
- Generates narrative explanations
- Identifies historical parallels
- Suggests potential interventions

### 5. Interactive Visualizations
- Real-time dashboard
- Historical charts with projections
- Scenario modeling
- Intervention simulations

## Configuration

Edit `config.py` to customize:
- Analysis parameters
- Time periods
- Thresholds and weights
- API model selections

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify virtual environment is activated

2. **API key errors**:
   - Check `.env` file exists and contains valid keys
   - Ensure no extra spaces around API keys

3. **Connection errors**:
   - Verify internet connection
   - Check API service status
   - Ensure API keys have sufficient credits

4. **Dashboard won't load**:
   - Check port 8050 is not in use
   - Try different port: `python dashboard.py --port 8051`

## Data Sources

The system uses:
- Historical economic data (simulated for demo)
- Political event timelines
- Social indicators
- LLM-enhanced historical analysis

## Output Files

The system generates:
- `output/analysis_report_YYYY-MM-DD.json` - Detailed analysis data
- `output/projections_YYYY-MM-DD.csv` - Future projections
- `output/report_YYYY-MM-DD.pdf` - Full PDF report (if requested)
- `output/visualizations/` - Chart images

## Contributing

Feel free to submit issues, feature requests, or pull requests.

## License

MIT License

## Disclaimer

This is a research tool for exploring cliodynamics concepts. Predictions should not be taken as definitive forecasts but rather as one lens for understanding complex social dynamics.

## Credits

Based on the structural-demographic theory developed by Peter Turchin and colleagues.