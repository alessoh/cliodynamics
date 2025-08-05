"""
Visualization module for creating charts and reports
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)

class Visualizer:
    """Creates visualizations for cliodynamics analysis"""
    
    def __init__(self, config):
        self.config = config
        
        # Set style
        plt.style.use(config.CHART_STYLE)
        sns.set_palette("husl")
        
        # Ensure output directories exist
        Path(config.VISUALIZATION_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.REPORT_DIR).mkdir(parents=True, exist_ok=True)
    
    def create_all_charts(self, indicators, cycles, projections):
        """Create all visualization charts"""
        logger.info("Creating visualizations...")
        
        charts = {}
        
        # Main PSI chart with projections
        charts['psi_chart'] = self._create_psi_chart(indicators, projections)
        
        # Multi-indicator comparison
        charts['indicators_chart'] = self._create_indicators_chart(indicators)
        
        # Cycle visualization
        charts['cycles_chart'] = self._create_cycles_chart(indicators, cycles)
        
        # Current state dashboard
        charts['dashboard'] = self._create_dashboard(indicators, cycles)
        
        # Projection scenarios
        charts['scenarios_chart'] = self._create_scenarios_chart(projections)
        
        # Historical events overlay
        charts['events_chart'] = self._create_events_chart(indicators)
        
        return charts
    
    def _create_psi_chart(self, indicators, projections):
        """Create main PSI chart with historical data and projections"""
        fig, ax = plt.subplots(figsize=self.config.CHART_SIZE, dpi=self.config.CHART_DPI)
        
        # Historical data
        hist_years = indicators['years']
        hist_psi = indicators['psi']
        
        # Projection data
        proj_years = projections['years']
        proj_psi = projections['psi']
        proj_lower = projections['confidence_intervals']['psi_lower']
        proj_upper = projections['confidence_intervals']['psi_upper']
        
        # Plot historical data
        ax.plot(hist_years, hist_psi, 'b-', linewidth=2, label='Historical PSI')
        
        # Plot projection
        ax.plot(proj_years, proj_psi, 'r--', linewidth=2, label='Projected PSI')
        
        # Confidence interval
        ax.fill_between(proj_years, proj_lower, proj_upper, 
                       color='red', alpha=0.2, label='Confidence Interval')
        
        # Add threshold lines
        ax.axhline(y=self.config.PSI_CRITICAL, color='darkred', linestyle=':', 
                  alpha=0.7, label=f'Critical ({self.config.PSI_CRITICAL})')
        ax.axhline(y=self.config.PSI_HIGH, color='orange', linestyle=':', 
                  alpha=0.7, label=f'High ({self.config.PSI_HIGH})')
        
        # Current year marker
        ax.axvline(x=self.config.CURRENT_YEAR, color='green', linestyle='-', 
                  alpha=0.7, label='Current Year')
        
        # Styling
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Political Stress Indicator (%)', fontsize=12)
        ax.set_title('Political Stress Indicator: Historical Trends and Projections', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim(0, 100)
        
        # Save
        filename = f"{self.config.VISUALIZATION_DIR}/psi_chart.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        logger.info(f"Saved PSI chart to {filename}")
        return filename
    
    def _save_figure(self, filename, dpi=None):
        """Safely save figure with fallback for large images"""
        if dpi is None:
            dpi = self.config.CHART_DPI
        
        try:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        except ValueError as e:
            if "too large" in str(e):
                logger.warning(f"Image too large at {dpi} DPI, reducing to 100 DPI")
                plt.savefig(filename, dpi=100, bbox_inches='tight')
            else:
                raise
    
    def _create_indicators_chart(self, indicators):
        """Create multi-indicator comparison chart"""
        # Limit the data points if too many years
        years = indicators['years']
        if len(years) > 200:
            # Sample every nth year to reduce data points
            step = len(years) // 200
            years = years[::step]
            indicators_sampled = {}
            for key, values in indicators.items():
                if key == 'years':
                    indicators_sampled[key] = years
                elif isinstance(values, list) and len(values) == len(indicators['years']):
                    indicators_sampled[key] = values[::step]
                else:
                    indicators_sampled[key] = values
            indicators = indicators_sampled
        
        fig, ax = plt.subplots(figsize=self.config.CHART_SIZE, dpi=self.config.CHART_DPI)
        
        years = indicators['years']
        
        # Plot all indicators
        ax.plot(years, indicators['psi'], label='PSI', linewidth=3, color='red')
        ax.plot(years, indicators['elite_overproduction'], 
               label='Elite Overproduction', linewidth=2, alpha=0.8)
        ax.plot(years, indicators['popular_immiseration'], 
               label='Popular Immiseration', linewidth=2, alpha=0.8)
        ax.plot(years, indicators['state_fiscal_health'], 
               label='State Fiscal Health', linewidth=2, alpha=0.8)
        ax.plot(years, indicators['intra_elite_competition'], 
               label='Intra-Elite Competition', linewidth=2, alpha=0.8)
        
        # Styling
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Indicator Value (%)', fontsize=12)
        ax.set_title('Structural-Demographic Indicators Over Time', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 100)
        
        # Save
        filename = f"{self.config.VISUALIZATION_DIR}/indicators_chart.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        return filename
    
    def _create_cycles_chart(self, indicators, cycles):
        """Create cycle visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.CHART_SIZE[0], 10), 
                                       dpi=self.config.CHART_DPI)
        
        years = indicators['years']
        psi = indicators['psi']
        
        # Top panel: PSI with cycle phases
        ax1.plot(years, psi, 'b-', linewidth=2)
        
        # Add secular cycle phases
        cycle_length = self.config.SECULAR_CYCLE_LENGTH
        for year in range(min(years), max(years), cycle_length):
            ax1.axvspan(year, year + cycle_length/4, alpha=0.2, color='green', 
                       label='Expansion' if year == min(years) else '')
            ax1.axvspan(year + cycle_length/4, year + cycle_length/2, 
                       alpha=0.2, color='yellow', label='Stagflation' if year == min(years) else '')
            ax1.axvspan(year + cycle_length/2, year + 3*cycle_length/4, 
                       alpha=0.2, color='orange', label='Crisis' if year == min(years) else '')
            ax1.axvspan(year + 3*cycle_length/4, year + cycle_length, 
                       alpha=0.2, color='gray', label='Depression' if year == min(years) else '')
        
        ax1.set_ylabel('PSI (%)', fontsize=12)
        ax1.set_title('Political Stress and Secular Cycles', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Bottom panel: Violence cycles
        violence_cycle = []
        for year in years:
            cycle_pos = ((year - 1870) % self.config.VIOLENCE_CYCLE_LENGTH) / self.config.VIOLENCE_CYCLE_LENGTH
            violence_cycle.append(30 + 30 * np.sin(cycle_pos * 2 * np.pi))
        
        ax2.plot(years, violence_cycle, 'r-', linewidth=2, label='50-Year Violence Cycle')
        ax2.plot(years, psi, 'b-', alpha=0.5, linewidth=1, label='PSI')
        
        # Mark violence peaks
        peaks = cycles.get('historical_peaks', [])
        for peak_year in peaks:
            if peak_year in years:
                ax2.axvline(x=peak_year, color='darkred', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Cycle Intensity', fontsize=12)
        ax2.set_title('Violence Cycles and Historical Peaks', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Save
        filename = f"{self.config.VISUALIZATION_DIR}/cycles_chart.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        return filename
    
    def _create_dashboard(self, indicators, cycles):
        """Create current state dashboard"""
        fig = plt.figure(figsize=(14, 10), dpi=self.config.CHART_DPI)
        
        # Current values
        current_idx = -1
        current_values = {
            'PSI': indicators['psi'][current_idx],
            'Elite Overproduction': indicators['elite_overproduction'][current_idx],
            'Popular Immiseration': indicators['popular_immiseration'][current_idx],
            'State Fiscal Health': indicators['state_fiscal_health'][current_idx],
            'Intra-Elite Competition': indicators['intra_elite_competition'][current_idx]
        }
        
        # Create gauge charts
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, (name, value) in enumerate(current_values.items()):
            ax = plt.subplot2grid((3, 3), positions[i])
            self._create_gauge(ax, name, value)
        
        # Cycle position
        ax_cycle = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1)
        self._create_cycle_position(ax_cycle, cycles)
        
        # Summary text
        ax_text = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax_text.axis('off')
        
        summary = f"Current Assessment: {cycles['secular_phase']} Phase\n"
        summary += f"PSI Trend: {cycles['trend_direction'].capitalize()}\n"
        summary += f"Years to Violence Peak: {cycles['years_to_violence_peak']}\n"
        summary += f"Overall Risk Level: {'CRITICAL' if current_values['PSI'] > 80 else 'HIGH' if current_values['PSI'] > 60 else 'MODERATE'}"
        
        ax_text.text(0.5, 0.5, summary, transform=ax_text.transAxes,
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Cliodynamics Dashboard - Current State', fontsize=16, fontweight='bold')
        
        # Save
        filename = f"{self.config.VISUALIZATION_DIR}/dashboard.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        return filename
    
    def _create_gauge(self, ax, label, value):
        """Create a gauge chart for a single indicator"""
        # Create semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'gray', linewidth=10, alpha=0.3)
        
        # Value arc
        value_theta = theta[0] + (theta[-1] - theta[0]) * value / 100
        value_arc = np.linspace(theta[0], value_theta, 50)
        
        # Color based on value
        if value < 30:
            color = 'green'
        elif value < 60:
            color = 'yellow'
        elif value < 80:
            color = 'orange'
        else:
            color = 'red'
        
        ax.plot(r * np.cos(value_arc), r * np.sin(value_arc), color, linewidth=10)
        
        # Needle
        ax.plot([0, r * np.cos(value_theta)], [0, r * np.sin(value_theta)], 
               'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Text
        ax.text(0, -0.3, f'{value:.1f}%', ha='center', va='center', fontsize=16, 
               fontweight='bold')
        ax.text(0, -0.6, label, ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.axis('off')
    
    def _create_cycle_position(self, ax, cycles):
        """Create cycle position indicator"""
        # Circular representation
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1
        
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'gray', linewidth=2)
        
        # Current position
        current_theta = 2 * np.pi * cycles['secular_position'] / 100
        ax.plot(r * np.cos(current_theta), r * np.sin(current_theta), 'ro', markersize=15)
        
        # Phase labels
        phases = ['Expansion', 'Stagflation', 'Crisis', 'Depression']
        phase_angles = [np.pi/2, 0, -np.pi/2, np.pi]
        
        for phase, angle in zip(phases, phase_angles):
            x = 1.3 * np.cos(angle)
            y = 1.3 * np.sin(angle)
            ax.text(x, y, phase, ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        ax.set_title('Secular Cycle Position', fontsize=12)
    
    def _create_scenarios_chart(self, projections):
        """Create scenario comparison chart"""
        # This would need multiple projection runs - simplified for demo
        fig, ax = plt.subplots(figsize=self.config.CHART_SIZE, dpi=self.config.CHART_DPI)
        
        years = projections['years']
        baseline = projections['psi']
        
        # Create synthetic optimistic and pessimistic scenarios
        optimistic = [p - 10 - i*0.5 for i, p in enumerate(baseline)]
        pessimistic = [p + 10 + i*0.5 for i, p in enumerate(baseline)]
        
        ax.plot(years, optimistic, 'g--', linewidth=2, label='Optimistic')
        ax.plot(years, baseline, 'b-', linewidth=3, label='Baseline')
        ax.plot(years, pessimistic, 'r--', linewidth=2, label='Pessimistic')
        
        # Fill between
        ax.fill_between(years, optimistic, pessimistic, alpha=0.2, color='gray')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('PSI (%)', fontsize=12)
        ax.set_title('Projection Scenarios', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        filename = f"{self.config.VISUALIZATION_DIR}/scenarios_chart.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        return filename
    
    def _create_events_chart(self, indicators):
        """Create chart with historical events overlay"""
        fig, ax = plt.subplots(figsize=self.config.CHART_SIZE, dpi=self.config.CHART_DPI)
        
        years = indicators['years']
        psi = indicators['psi']
        
        ax.plot(years, psi, 'b-', linewidth=2, label='PSI')
        
        # Add event markers
        for start, end, event in self.config.MAJOR_CRISIS_YEARS:
            if start in years:
                ax.axvspan(start, end, alpha=0.3, color='red')
                ax.text(start, 90, event, rotation=90, fontsize=8, 
                       va='bottom', ha='right')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('PSI (%)', fontsize=12)
        ax.set_title('Political Stress and Major Historical Events', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        filename = f"{self.config.VISUALIZATION_DIR}/events_chart.png"
        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
        
        return filename
    
    def generate_pdf_report(self, analysis_results):
        """Generate comprehensive PDF report"""
        logger.info("Generating PDF report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.config.REPORT_DIR}/cliodynamics_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph("Cliodynamics Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        story.append(Paragraph(f"Analysis Date: {analysis_results['metadata']['analysis_date']}", 
                             styles['Normal']))
        story.append(Paragraph(f"Period: {analysis_results['metadata']['start_year']} - "
                             f"{analysis_results['metadata']['end_year']}", styles['Normal']))
        story.append(Paragraph(f"Scenario: {analysis_results['metadata']['scenario'].capitalize()}", 
                             styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Paragraph(analysis_results['llm_insights']['summary'], styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Current State
        story.append(Paragraph("Current State Assessment", styles['Heading2']))
        
        current_data = [
            ['Indicator', 'Value', 'Status'],
            ['Political Stress Indicator', 
             f"{analysis_results['current_state']['psi']:.1f}%",
             self._get_status(analysis_results['current_state']['psi'])],
            ['Elite Overproduction', 
             f"{analysis_results['current_state']['elite_overproduction']:.1f}%",
             self._get_status(analysis_results['current_state']['elite_overproduction'])],
            ['Popular Immiseration', 
             f"{analysis_results['current_state']['popular_immiseration']:.1f}%",
             self._get_status(analysis_results['current_state']['popular_immiseration'])],
            ['State Fiscal Health', 
             f"{analysis_results['current_state']['state_fiscal_health']:.1f}%",
             self._get_status(100 - analysis_results['current_state']['state_fiscal_health'])],
            ['Intra-Elite Competition', 
             f"{analysis_results['current_state']['intra_elite_competition']:.1f}%",
             self._get_status(analysis_results['current_state']['intra_elite_competition'])]
        ]
        
        t = Table(current_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(PageBreak())
        
        # Visualizations
        story.append(Paragraph("Key Visualizations", styles['Heading2']))
        
        # Add charts
        for chart_name, chart_path in analysis_results['visualizations'].items():
            if Path(chart_path).exists():
                img = Image(chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        
        for rec in analysis_results['llm_insights']['recommendations']:
            story.append(Paragraph(f"<b>{rec['area']}</b> (Priority: {rec['priority']})", 
                                 styles['Normal']))
            story.append(Paragraph(rec['recommendation'], styles['Normal']))
            
            for action in rec['specific_actions']:
                story.append(Paragraph(f"• {action}", styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report saved to {filename}")
        return filename
    
    def _get_status(self, value):
        """Get status label for indicator value"""
        if value < 30:
            return "Low"
        elif value < 60:
            return "Moderate"
        elif value < 80:
            return "High"
        else:
            return "Critical"