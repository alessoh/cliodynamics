#!/usr/bin/env python3
"""
Interactive web dashboard for Cliodynamics Analysis
Built with Plotly Dash for real-time exploration
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging

# Project imports
from config import Config
from cliodynamics_core import CliodynamicsEngine
from data_collector import DataCollector
from llm_analyzer import LLMAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Cliodynamics Analysis Dashboard"

# Initialize components
data_collector = DataCollector(config)
engine = CliodynamicsEngine(config)

# Load most recent analysis data
def load_latest_analysis():
    """Load the most recent analysis results"""
    output_dir = Path("output")
    files = list(output_dir.glob("analysis_report_*.json"))
    
    if not files:
        # Generate default data if no analysis exists
        logger.warning("No analysis files found, generating default data")
        historical_data = data_collector.collect_historical_data(1950, 2024)
        indicators = engine.calculate_indicators(historical_data)
        cycles = engine.identify_cycles(indicators)
        projections = engine.generate_projections(indicators, cycles)
        
        return {
            'indicators': indicators,
            'cycles': cycles,
            'projections': projections,
            'historical_data': historical_data
        }
    
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data

# Load initial data
analysis_data = load_latest_analysis()

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Cliodynamics Analysis Dashboard", 
                   className="text-center mb-4",
                   style={'color': '#4fc3f7'}),
            html.P("Monitoring Humanity's Position in Long-Term Social Cycles", 
                  className="text-center text-muted mb-4")
        ])
    ]),
    
    # Current State Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Political Stress Index", className="card-title"),
                    html.H2(id="psi-current", className="text-center"),
                    dcc.Graph(id="psi-gauge", style={'height': '200px'})
                ])
            ], color="danger" if analysis_data.get('current_state', {}).get('psi', 0) > 60 else "warning")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Secular Cycle Phase", className="card-title"),
                    html.H3(id="cycle-phase", className="text-center mt-3"),
                    html.P(id="cycle-description", className="text-center")
                ])
            ], color="info")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Years to Violence Peak", className="card-title"),
                    html.H2(id="violence-peak", className="text-center mt-3"),
                    html.P("Based on 50-year cycles", className="text-center text-muted")
                ])
            ], color="warning")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Risk Level", className="card-title"),
                    html.H3(id="risk-level", className="text-center mt-3"),
                    dcc.Graph(id="risk-indicator", style={'height': '150px'})
                ])
            ], color="danger")
        ], md=3)
    ], className="mb-4"),
    
    # Main Chart Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Historical Analysis", className="mb-0")
                        ], md=6),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("PSI", id="btn-psi", color="primary", size="sm"),
                                dbc.Button("All Indicators", id="btn-all", color="secondary", size="sm"),
                                dbc.Button("Cycles", id="btn-cycles", color="secondary", size="sm"),
                                dbc.Button("Events", id="btn-events", color="secondary", size="sm")
                            ])
                        ], md=6, className="text-right")
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id="main-chart", style={'height': '500px'})
                ])
            ])
        ], md=12)
    ], className="mb-4"),
    
    # Controls and Projections
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Scenario Projections"),
                dbc.CardBody([
                    dcc.RadioItems(
                        id="scenario-selector",
                        options=[
                            {'label': 'Optimistic', 'value': 'optimistic'},
                            {'label': 'Baseline', 'value': 'baseline'},
                            {'label': 'Pessimistic', 'value': 'pessimistic'}
                        ],
                        value='baseline',
                        inline=True,
                        className="mb-3"
                    ),
                    dcc.Graph(id="projection-chart", style={'height': '300px'})
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Key Indicators"),
                dbc.CardBody([
                    dcc.Graph(id="indicators-radar", style={'height': '300px'})
                ])
            ])
        ], md=6)
    ], className="mb-4"),
    
    # Detailed Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Analysis & Insights"),
                dbc.CardBody([
                    html.Div(id="ai-insights", className="p-3")
                ])
            ])
        ], md=12)
    ], className="mb-4"),
    
    # Refresh interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute
        n_intervals=0
    ),
    
    # Store component for data
    dcc.Store(id='analysis-data-store', data=analysis_data)
    
], fluid=True, className="p-4")

# Callbacks
@app.callback(
    [Output('psi-current', 'children'),
     Output('cycle-phase', 'children'),
     Output('cycle-description', 'children'),
     Output('violence-peak', 'children'),
     Output('risk-level', 'children')],
    [Input('analysis-data-store', 'data')]
)
def update_current_state(data):
    """Update current state displays"""
    if not data or 'current_state' not in data:
        return "N/A", "N/A", "", "N/A", "N/A"
    
    current = data.get('current_state', {})
    cycles = data.get('cycles', {})
    
    psi = f"{current.get('psi', 0):.1f}%"
    phase = cycles.get('secular_phase', 'Unknown')
    
    phase_descriptions = {
        'Expansion': 'Growth and opportunity phase',
        'Stagflation': 'Slowing growth, rising tensions',
        'Crisis': 'High instability and conflict',
        'Depression': 'Recovery and reorganization'
    }
    
    description = phase_descriptions.get(phase, '')
    violence_peak = f"{cycles.get('years_to_violence_peak', 'N/A')} years"
    
    # Determine risk level
    psi_value = current.get('psi', 0)
    if psi_value > 80:
        risk = "CRITICAL"
    elif psi_value > 60:
        risk = "HIGH"
    elif psi_value > 40:
        risk = "MODERATE"
    else:
        risk = "LOW"
    
    return psi, phase, description, violence_peak, risk

@app.callback(
    Output('psi-gauge', 'figure'),
    [Input('analysis-data-store', 'data')]
)
def update_psi_gauge(data):
    """Update PSI gauge chart"""
    if not data or 'current_state' not in data:
        return go.Figure()
    
    psi_value = data['current_state'].get('psi', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=psi_value,
        title={'text': ""},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if psi_value > 80 else "orange" if psi_value > 60 else "yellow"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 80], 'color': "darkgray"},
                {'range': [80, 100], 'color': "black"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig

@app.callback(
    Output('main-chart', 'figure'),
    [Input('btn-psi', 'n_clicks'),
     Input('btn-all', 'n_clicks'),
     Input('btn-cycles', 'n_clicks'),
     Input('btn-events', 'n_clicks')],
    [State('analysis-data-store', 'data')]
)
def update_main_chart(btn_psi, btn_all, btn_cycles, btn_events, data):
    """Update main chart based on selection"""
    if not data:
        return go.Figure()
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'btn-psi'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    indicators = data.get('indicators', {})
    years = indicators.get('years', [])
    
    fig = go.Figure()
    
    if button_id == 'btn-psi' or button_id == 'btn-events':
        # PSI with projections
        fig.add_trace(go.Scatter(
            x=years,
            y=indicators.get('psi', []),
            mode='lines',
            name='Historical PSI',
            line=dict(color='#4fc3f7', width=3)
        ))
        
        # Add projections if available
        if 'projections' in data:
            proj = data['projections']
            fig.add_trace(go.Scatter(
                x=proj.get('years', []),
                y=proj.get('psi', []),
                mode='lines',
                name='Projected PSI',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Add events if selected
        if button_id == 'btn-events':
            for start, end, event in config.MAJOR_CRISIS_YEARS:
                if start in years:
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="red", opacity=0.2,
                        layer="below", line_width=0,
                        annotation_text=event,
                        annotation_position="top left"
                    )
    
    elif button_id == 'btn-all':
        # All indicators
        for indicator in ['psi', 'elite_overproduction', 'popular_immiseration', 
                         'state_fiscal_health', 'intra_elite_competition']:
            if indicator in indicators:
                fig.add_trace(go.Scatter(
                    x=years,
                    y=indicators[indicator],
                    mode='lines',
                    name=indicator.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
    
    elif button_id == 'btn-cycles':
        # Cycles visualization
        fig.add_trace(go.Scatter(
            x=years,
            y=indicators.get('psi', []),
            mode='lines',
            name='PSI',
            line=dict(color='#4fc3f7', width=2)
        ))
        
        # Add cycle phases as background
        cycle_length = config.SECULAR_CYCLE_LENGTH
        for year in range(min(years), max(years), cycle_length):
            colors = ['rgba(0,255,0,0.1)', 'rgba(255,255,0,0.1)', 
                     'rgba(255,165,0,0.1)', 'rgba(128,128,128,0.1)']
            phases = ['Expansion', 'Stagflation', 'Crisis', 'Depression']
            
            for i, (color, phase) in enumerate(zip(colors, phases)):
                fig.add_vrect(
                    x0=year + i*cycle_length/4,
                    x1=year + (i+1)*cycle_length/4,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    annotation_text=phase if year == min(years) else "",
                    annotation_position="top left"
                )
    
    # Update layout
    fig.update_layout(
        title="Historical Patterns and Projections",
        xaxis_title="Year",
        yaxis_title="Value (%)",
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    # Add current year line
    fig.add_vline(x=config.CURRENT_YEAR, line_dash="dash", 
                 line_color="green", opacity=0.5)
    
    return fig

@app.callback(
    Output('projection-chart', 'figure'),
    [Input('scenario-selector', 'value')],
    [State('analysis-data-store', 'data')]
)
def update_projection_chart(scenario, data):
    """Update projection chart based on scenario"""
    if not data or 'projections' not in data:
        return go.Figure()
    
    proj = data['projections']
    years = proj.get('years', [])
    psi = proj.get('psi', [])
    
    # Create synthetic scenarios (in real app, would recalculate)
    if scenario == 'optimistic':
        psi_scenario = [p - 10 - i*0.3 for i, p in enumerate(psi)]
    elif scenario == 'pessimistic':
        psi_scenario = [p + 10 + i*0.3 for i, p in enumerate(psi)]
    else:
        psi_scenario = psi
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=psi_scenario,
        mode='lines',
        name=f'{scenario.capitalize()} Scenario',
        line=dict(width=3)
    ))
    
    # Add confidence band
    upper = [min(100, p + 10) for p in psi_scenario]
    lower = [max(0, p - 10) for p in psi_scenario]
    
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(255,255,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Confidence Band'
    ))
    
    fig.update_layout(
        title=f"PSI Projection - {scenario.capitalize()} Scenario",
        xaxis_title="Year",
        yaxis_title="PSI (%)",
        template='plotly_dark',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

@app.callback(
    Output('indicators-radar', 'figure'),
    [Input('analysis-data-store', 'data')]
)
def update_radar_chart(data):
    """Update radar chart of current indicators"""
    if not data or 'current_state' not in data:
        return go.Figure()
    
    current = data['current_state']
    
    categories = ['PSI', 'Elite\nOverproduction', 'Popular\nImmiseration', 
                 'State Fiscal\nHealth', 'Elite\nCompetition']
    
    values = [
        current.get('psi', 0),
        current.get('elite_overproduction', 0),
        current.get('popular_immiseration', 0),
        100 - current.get('state_fiscal_health', 0),  # Invert for stress
        current.get('intra_elite_competition', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current State',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2)
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatterpolar(
        r=[60] * len(categories),
        theta=categories,
        mode='lines',
        name='High Threshold',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        template='plotly_dark',
        height=300,
        showlegend=True
    )
    
    return fig

@app.callback(
    Output('ai-insights', 'children'),
    [Input('analysis-data-store', 'data')]
)
def update_ai_insights(data):
    """Update AI insights section"""
    if not data or 'llm_insights' not in data:
        return html.P("No AI insights available")
    
    insights = data['llm_insights']
    
    content = [
        html.H5("Summary", className="mb-3"),
        html.P(insights.get('summary', 'No summary available')),
        
        html.H5("Key Risk Factors", className="mb-3"),
        html.Ul([
            html.Li(risk) for risk in insights.get('risk_factors', [])[:5]
        ]),
        
        html.H5("Recommendations", className="mb-3")
    ]
    
    # Add recommendations
    for rec in insights.get('recommendations', [])[:3]:
        content.extend([
            html.H6(f"{rec.get('area', 'Unknown')} - Priority: {rec.get('priority', 'Unknown')}"),
            html.P(rec.get('recommendation', '')),
            html.Ul([
                html.Li(action) for action in rec.get('specific_actions', [])[:3]
            ], className="mb-3")
        ])
    
    return content

@app.callback(
    Output('risk-indicator', 'figure'),
    [Input('analysis-data-store', 'data')]
)
def update_risk_indicator(data):
    """Update risk indicator mini chart"""
    if not data:
        return go.Figure()
    
    # Simple risk score calculation
    current = data.get('current_state', {})
    psi = current.get('psi', 0)
    
    risk_score = min(100, psi * 1.2)  # Amplify for visualization
    
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=risk_score,
        delta={'reference': 60, 'relative': False},
        number={'suffix': "%"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'size': 20}
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    logger.info(f"Starting dashboard on http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run_server(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG
    )