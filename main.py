#!/usr/bin/env python3
"""
AI Cliodynamics Analysis System - Main Entry Point
Analyzes historical patterns and predicts social cycles using LLMs
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Project imports
from config import Config
from cliodynamics_core import CliodynamicsEngine
from llm_analyzer import LLMAnalyzer
from data_collector import DataCollector
from visualizer import Visualizer
from utils import setup_logging, create_output_dirs, format_analysis_report

# Setup logging
logger = setup_logging()

class CliodynamicsAnalysisSystem:
    """Main system orchestrator for cliodynamics analysis"""
    
    def __init__(self, config=None):
        """Initialize the analysis system"""
        self.config = config or Config()
        create_output_dirs()
        
        # Initialize components
        logger.info("Initializing Cliodynamics Analysis System...")
        self.data_collector = DataCollector(self.config)
        self.engine = CliodynamicsEngine(self.config)
        self.llm_analyzer = LLMAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        
    def run_analysis(self, start_year=None, end_year=None, scenario='baseline'):
        """Run complete cliodynamics analysis"""
        try:
            # Set analysis period
            start_year = start_year or self.config.DEFAULT_START_YEAR
            end_year = end_year or self.config.DEFAULT_END_YEAR
            
            logger.info(f"Running analysis from {start_year} to {end_year} with {scenario} scenario")
            
            # Step 1: Collect and prepare data
            logger.info("Collecting historical data...")
            historical_data = self.data_collector.collect_historical_data(start_year, end_year)
            
            # Step 2: Calculate cliodynamics indicators
            logger.info("Calculating structural-demographic indicators...")
            indicators = self.engine.calculate_indicators(historical_data)
            
            # Step 3: Identify cycles and patterns
            logger.info("Identifying secular and violence cycles...")
            cycles = self.engine.identify_cycles(indicators)
            
            # Step 4: Get LLM analysis
            logger.info("Generating AI-enhanced analysis...")
            llm_insights = self.llm_analyzer.analyze_patterns(
                indicators, 
                cycles, 
                historical_data
            )
            
            # Step 5: Generate projections
            logger.info(f"Generating {scenario} projections...")
            projections = self.engine.generate_projections(
                indicators, 
                cycles, 
                scenario=scenario
            )
            
            # Step 6: Create visualizations
            logger.info("Creating visualizations...")
            charts = self.visualizer.create_all_charts(
                indicators, 
                cycles, 
                projections
            )
            
            # Step 7: Compile results
            analysis_results = {
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'start_year': start_year,
                    'end_year': end_year,
                    'scenario': scenario
                },
                'current_state': {
                    'psi': indicators['psi'][-1],
                    'elite_overproduction': indicators['elite_overproduction'][-1],
                    'popular_immiseration': indicators['popular_immiseration'][-1],
                    'state_fiscal_health': indicators['state_fiscal_health'][-1],
                    'intra_elite_competition': indicators['intra_elite_competition'][-1]
                },
                'cycles': cycles,
                'projections': projections,
                'llm_insights': llm_insights,
                'visualizations': charts
            }
            
            # Save results
            self._save_results(analysis_results)
            
            # Display summary
            self._display_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def generate_report(self, analysis_results=None):
        """Generate comprehensive PDF report"""
        if not analysis_results:
            # Load most recent analysis
            analysis_results = self._load_latest_results()
            
        logger.info("Generating PDF report...")
        report_path = self.visualizer.generate_pdf_report(analysis_results)
        logger.info(f"Report saved to: {report_path}")
        return report_path
    
    def _save_results(self, results):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"output/analysis_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Results saved to {filename}")
        
    def _load_latest_results(self):
        """Load most recent analysis results"""
        output_dir = Path("output")
        files = list(output_dir.glob("analysis_report_*.json"))
        if not files:
            raise ValueError("No previous analysis results found")
            
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def _display_summary(self, results):
        """Display analysis summary to console"""
        print("\n" + "="*60)
        print("CLIODYNAMICS ANALYSIS SUMMARY")
        print("="*60)
        
        current = results['current_state']
        print(f"\nCurrent State Assessment:")
        print(f"  Political Stress Indicator: {current['psi']:.1f}%")
        print(f"  Elite Overproduction: {current['elite_overproduction']:.1f}%")
        print(f"  Popular Immiseration: {current['popular_immiseration']:.1f}%")
        print(f"  State Fiscal Health: {current['state_fiscal_health']:.1f}%")
        print(f"  Intra-Elite Competition: {current['intra_elite_competition']:.1f}%")
        
        cycles = results['cycles']
        print(f"\nCycle Position:")
        print(f"  Secular Cycle Phase: {cycles['secular_phase']}")
        print(f"  Years to Next Violence Peak: {cycles['years_to_violence_peak']}")
        
        print(f"\nAI Analysis Summary:")
        print(f"  {results['llm_insights']['summary'][:200]}...")
        
        print("\n" + "="*60)
        print("Analysis complete. See output folder for detailed results.")
        print("="*60 + "\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Cliodynamics Analysis System"
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        help='Start year for analysis (default: 1800)'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        help='End year for analysis (default: current + 30)'
    )
    parser.add_argument(
        '--scenario', 
        choices=['optimistic', 'baseline', 'pessimistic'],
        default='baseline',
        help='Projection scenario (default: baseline)'
    )
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Generate PDF report after analysis'
    )
    parser.add_argument(
        '--dashboard', 
        action='store_true',
        help='Launch interactive dashboard after analysis'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = CliodynamicsAnalysisSystem()
        
        # Run analysis
        results = system.run_analysis(
            start_year=args.start_year,
            end_year=args.end_year,
            scenario=args.scenario
        )
        
        # Generate report if requested
        if args.report:
            system.generate_report(results)
            
        # Launch dashboard if requested
        if args.dashboard:
            logger.info("Launching dashboard...")
            import subprocess
            subprocess.Popen([sys.executable, 'dashboard.py'])
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()