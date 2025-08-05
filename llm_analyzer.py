"""
LLM integration for enhanced historical analysis
Uses OpenAI and Anthropic APIs for pattern recognition and insights
"""

import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import os

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Integrates LLMs for enhanced cliodynamics analysis"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize Anthropic client
        try:
            # Get the API key and ensure it's clean
            api_key = config.ANTHROPIC_API_KEY
            if api_key:
                api_key = api_key.strip()
                
            logger.info(f"Initializing Anthropic client with model: {config.ANTHROPIC_MODEL}")
            
            self.anthropic_client = anthropic.Anthropic(
                api_key=api_key
            )
            
            # Test the client with a minimal call
            try:
                test_response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                logger.info("Anthropic client test successful")
                self.use_claude = True
            except Exception as e:
                logger.error(f"Anthropic client test failed: {str(e)}")
                self.use_claude = False
                
        except Exception as e:
            logger.warning(f"Claude API not available: {str(e)}. Running with OpenAI only.")
            self.anthropic_client = None
            self.use_claude = False
        
        logger.info(f"LLM Analyzer initialized (Claude available: {self.use_claude})")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_patterns(self, indicators: Dict, cycles: Dict, historical_data: Dict) -> Dict:
        """Use LLMs to analyze historical patterns and generate insights"""
        logger.info("Generating AI-enhanced analysis...")
        
        # Prepare context for LLMs
        context = self._prepare_context(indicators, cycles, historical_data)
        
        # Get analysis from both LLMs
        openai_analysis = self._get_openai_analysis(context)
        
        # Only get Claude analysis if available
        if self.use_claude:
            claude_analysis = self._get_claude_analysis(context)
        else:
            claude_analysis = self._get_fallback_analysis("claude")
        
        # Synthesize insights
        synthesized = self._synthesize_insights(openai_analysis, claude_analysis)
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(synthesized, indicators)
        
        return {
            'summary': synthesized['summary'],
            'historical_parallels': synthesized['parallels'],
            'key_drivers': synthesized['drivers'],
            'risk_factors': synthesized['risks'],
            'recommendations': recommendations,
            'detailed_analysis': {
                'openai': openai_analysis,
                'claude': claude_analysis
            }
        }
    
    def _prepare_context(self, indicators: Dict, cycles: Dict, historical_data: Dict) -> str:
        """Prepare context for LLM analysis"""
        current_year = self.config.CURRENT_YEAR
        current_idx = -1  # Last value is current
        
        # Identify relevant historical events
        historical_events = []
        for start, end, event in self.config.MAJOR_CRISIS_YEARS:
            if start >= current_year - 50:  # Recent history
                historical_events.append(f"{start}-{end}: {event}")
        
        context = self.config.HISTORICAL_ANALYSIS_PROMPT.format(
            start_year=indicators['years'][0],
            end_year=indicators['years'][-1],
            psi=f"{indicators['psi'][current_idx]:.1f}",
            elite=f"{indicators['elite_overproduction'][current_idx]:.1f}",
            immiseration=f"{indicators['popular_immiseration'][current_idx]:.1f}",
            fiscal=f"{indicators['state_fiscal_health'][current_idx]:.1f}",
            phase=cycles['secular_phase'],
            historical_events="\n".join(historical_events)
        )
        
        # Add trend information
        context += f"\n\nRecent Trends:\n"
        context += f"PSI trend: {cycles['trend_direction']}\n"
        context += f"Years to next violence cycle peak: {cycles['years_to_violence_peak']}\n"
        
        return context
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_openai_analysis(self, context: str) -> Dict:
        """Get analysis from OpenAI"""
        logger.info("Querying OpenAI for analysis...")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in cliodynamics and structural-demographic theory. Analyze historical patterns and provide insights based on quantitative historical analysis."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse structured response
            return self._parse_llm_response(analysis_text, "openai")
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return self._get_fallback_analysis("openai")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_claude_analysis(self, context: str) -> Dict:
        """Get analysis from Claude"""
        logger.info("Querying Claude for analysis...")
        
        try:
            # Debug: Log API key format (safely)
            api_key = self.config.ANTHROPIC_API_KEY.strip()
            logger.debug(f"API key format check: starts with 'sk-ant-api' = {api_key.startswith('sk-ant-api')}")
            
            response = self.anthropic_client.messages.create(
                model=self.config.ANTHROPIC_MODEL,
                max_tokens=1500,
                temperature=0.7,
                system="You are an expert in cliodynamics and structural-demographic theory. Analyze historical patterns and provide insights based on quantitative historical analysis.",
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )
            
            analysis_text = response.content[0].text
            
            # Parse structured response
            return self._parse_llm_response(analysis_text, "claude")
            
        except anthropic.AuthenticationError as e:
            logger.error(f"Claude Authentication Error: {str(e)}")
            logger.error("Please check that your ANTHROPIC_API_KEY in .env is correct")
            logger.error("The key should start with 'sk-ant-api03-' or similar")
            return self._get_fallback_analysis("claude")
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return self._get_fallback_analysis("claude")
    
    def _parse_llm_response(self, response_text: str, source: str) -> Dict:
        """Parse LLM response into structured format"""
        # This is a simplified parser - in production would use more sophisticated NLP
        
        sections = {
            'parallels': [],
            'drivers': [],
            'tipping_points': [],
            'interventions': [],
            'scenarios': []
        }
        
        current_section = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'historical parallel' in line.lower() or 'parallel' in line.lower():
                current_section = 'parallels'
            elif 'driver' in line.lower() or 'cause' in line.lower():
                current_section = 'drivers'
            elif 'tipping point' in line.lower() or 'threshold' in line.lower():
                current_section = 'tipping_points'
            elif 'intervention' in line.lower() or 'recommendation' in line.lower():
                current_section = 'interventions'
            elif 'scenario' in line.lower() or 'projection' in line.lower():
                current_section = 'scenarios'
            elif current_section and line.startswith(('•', '-', '*', '1', '2', '3', '4', '5')):
                # Extract bullet points
                clean_line = line.lstrip('•-*0123456789. ')
                if clean_line:
                    sections[current_section].append(clean_line)
        
        return {
            'source': source,
            'full_text': response_text,
            'structured': sections
        }
    
    def _synthesize_insights(self, openai_analysis: Dict, claude_analysis: Dict) -> Dict:
        """Synthesize insights from multiple LLMs"""
        # Combine insights from both sources
        combined = {
            'parallels': [],
            'drivers': [],
            'risks': [],
            'summary': ""
        }
        
        # Merge parallels (deduplicate similar ones)
        all_parallels = (openai_analysis.get('structured', {}).get('parallels', []) + 
                        claude_analysis.get('structured', {}).get('parallels', []))
        combined['parallels'] = list(set(all_parallels))[:5]  # Top 5 unique
        
        # Merge drivers
        all_drivers = (openai_analysis.get('structured', {}).get('drivers', []) + 
                      claude_analysis.get('structured', {}).get('drivers', []))
        combined['drivers'] = list(set(all_drivers))[:5]
        
        # Merge risks (tipping points)
        all_risks = (openai_analysis.get('structured', {}).get('tipping_points', []) + 
                    claude_analysis.get('structured', {}).get('tipping_points', []))
        combined['risks'] = list(set(all_risks))[:5]
        
        # Create summary
        if combined['parallels']:
            combined['summary'] = f"Current conditions show parallels to {combined['parallels'][0]}. "
        combined['summary'] += f"Key drivers include {', '.join(combined['drivers'][:2]) if combined['drivers'] else 'multiple socioeconomic factors'}. "
        combined['summary'] += "Structural-demographic analysis indicates elevated risk levels requiring attention."
        
        return combined
    
    def _generate_recommendations(self, synthesized: Dict, indicators: Dict) -> List[Dict]:
        """Generate specific policy recommendations"""
        recommendations = []
        
        current_psi = indicators['psi'][-1]
        current_elite = indicators['elite_overproduction'][-1]
        current_immiseration = indicators['popular_immiseration'][-1]
        current_fiscal = indicators['state_fiscal_health'][-1]
        
        # Priority-based recommendations
        if current_elite > self.config.PSI_CRITICAL:
            recommendations.append({
                'priority': 'Critical',
                'area': 'Elite Overproduction',
                'recommendation': 'Urgent reforms needed in higher education and elite pathways',
                'specific_actions': [
                    'Expand alternative career paths for graduates',
                    'Reform credential inflation in professional sectors',
                    'Create new high-status positions in emerging fields'
                ]
            })
        
        if current_immiseration > self.config.PSI_HIGH:
            recommendations.append({
                'priority': 'High',
                'area': 'Popular Immiseration',
                'recommendation': 'Address economic inequality and living standards',
                'specific_actions': [
                    'Increase minimum wage indexed to productivity',
                    'Expand access to healthcare and education',
                    'Strengthen labor protections and bargaining power'
                ]
            })
        
        if current_fiscal < 40:  # Low fiscal health
            recommendations.append({
                'priority': 'High',
                'area': 'State Fiscal Health',
                'recommendation': 'Restore fiscal capacity and public trust',
                'specific_actions': [
                    'Progressive tax reform to increase revenue',
                    'Reduce wasteful spending and corruption',
                    'Invest in productivity-enhancing infrastructure'
                ]
            })
        
        # Always include monitoring recommendation
        recommendations.append({
            'priority': 'Ongoing',
            'area': 'Monitoring and Early Warning',
            'recommendation': 'Establish systematic monitoring of structural-demographic indicators',
            'specific_actions': [
                'Create interdisciplinary task force for social stability',
                'Develop real-time dashboard for key indicators',
                'Regular scenario planning and stress testing'
            ]
        })
        
        return recommendations
    
    def _get_fallback_analysis(self, source: str) -> Dict:
        """Provide fallback analysis if API fails"""
        return {
            'source': source,
            'full_text': "API temporarily unavailable. Using fallback analysis based on indicators.",
            'structured': {
                'parallels': ["Similar to pre-crisis periods in historical cycles"],
                'drivers': ["Elite overproduction", "Economic inequality"],
                'tipping_points': ["PSI exceeding 80%"],
                'interventions': ["Address structural imbalances"],
                'scenarios': ["Baseline projection shows continued stress"]
            }
        }
    
    def analyze_projections(self, projections: Dict, scenario: str) -> Dict:
        """Analyze projection scenarios using LLMs"""
        context = self.config.PROJECTION_ANALYSIS_PROMPT.format(
            scenario=scenario,
            projection_data=json.dumps({
                'final_psi': projections['psi'][-1],
                'psi_change': projections['psi'][-1] - projections['psi'][0],
                'years': len(projections['years']),
                'key_indicators': {
                    'elite_overproduction': projections['elite_overproduction'][-1],
                    'popular_immiseration': projections['popular_immiseration'][-1],
                    'state_fiscal_health': projections['state_fiscal_health'][-1]
                }
            }, indent=2)
        )
        
        # For brevity, just use OpenAI for projection analysis
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze cliodynamics projections and assess likelihood and risks."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return {
                'scenario': scenario,
                'analysis': response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error(f"Projection analysis error: {str(e)}")
            return {
                'scenario': scenario,
                'analysis': f"{scenario.capitalize()} scenario shows projected trends based on current trajectories."
            }