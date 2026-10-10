import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class UncertaintyAwareScaler:
    """Implements uncertainty-aware adaptive scaling strategies."""
    
    def __init__(self, base_strategies: Dict[str, Dict]):
        """Initialize with base scaling strategies for each archetype.
        
        Args:
            base_strategies: Dictionary mapping archetype to strategy parameters
        """
        self.base_strategies = base_strategies
        self.uncertainty_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
    def compute_uncertainty_adjusted_params(self, 
                                          archetype: str, 
                                          confidence: float,
                                          current_load: float) -> Dict:
        """Compute scaling parameters adjusted by prediction uncertainty.
        
        Args:
            archetype: Predicted workload archetype
            confidence: Confidence score of the prediction (0-1)
            current_load: Current system load/utilization
            
        Returns:
            Adjusted scaling parameters
        """
        base_params = self.base_strategies.get(archetype, self.base_strategies['STATIONARY_NOISY'])
        
        # Determine uncertainty level
        if confidence >= self.uncertainty_thresholds['high_confidence']:
            uncertainty_level = 'low'
            adjustment_factor = 1.0  # Full confidence in prediction
        elif confidence >= self.uncertainty_thresholds['medium_confidence']:
            uncertainty_level = 'medium'
            adjustment_factor = 0.7  # Moderate adjustment
        else:
            uncertainty_level = 'high'
            adjustment_factor = 0.4  # Conservative adjustment
            
        # Adjust parameters based on uncertainty
        adjusted_params = self._apply_uncertainty_adjustment(
            base_params.copy(), 
            adjustment_factor,
            uncertainty_level,
            archetype,
            current_load
        )
        
        adjusted_params['confidence'] = confidence
        adjusted_params['uncertainty_level'] = uncertainty_level
        
        return adjusted_params
    
    def _apply_uncertainty_adjustment(self, 
                                    params: Dict, 
                                    adjustment_factor: float,
                                    uncertainty_level: str,
                                    archetype: str,
                                    current_load: float) -> Dict:
        """Apply uncertainty-based adjustments to scaling parameters.
        
        High uncertainty -> More conservative scaling
        Low uncertainty -> More aggressive scaling
        """
        
        # Adjust target utilization
        if 'target_cpu_utilization' in params:
            if uncertainty_level == 'high':
                # Lower target = more headroom for uncertainty
                params['target_cpu_utilization'] *= 0.8
            elif uncertainty_level == 'medium':
                params['target_cpu_utilization'] *= 0.9
            # else: keep original for low uncertainty
        
        # Adjust cooldown periods
        if 'scale_down_cooldown' in params:
            if uncertainty_level == 'high':
                # Longer cooldown = more stability
                params['scale_down_cooldown'] = int(params['scale_down_cooldown'] * 1.5)
            elif uncertainty_level == 'medium':
                params['scale_down_cooldown'] = int(params['scale_down_cooldown'] * 1.2)
        
        # Archetype-specific adjustments
        if archetype == 'SPIKE':
            params = self._adjust_spike_params(params, uncertainty_level, current_load)
        elif archetype == 'PERIODIC':
            params = self._adjust_periodic_params(params, uncertainty_level)
        elif archetype == 'RAMP':
            params = self._adjust_ramp_params(params, uncertainty_level)
            
        # Add safety margins based on uncertainty
        params['safety_margin'] = self._compute_safety_margin(uncertainty_level, archetype)
        
        return params
    
    def _adjust_spike_params(self, params: Dict, uncertainty_level: str, current_load: float) -> Dict:
        """Adjust spike-specific parameters based on uncertainty."""
        if uncertainty_level == 'high':
            # Very conservative for uncertain spikes
            params['spike_buffer_multiplier'] = 2.0  # Double buffer
            params['min_replicas'] = max(2, int(params.get('min_replicas', 1) * 1.5))
        elif uncertainty_level == 'medium':
            params['spike_buffer_multiplier'] = 1.5
            params['min_replicas'] = max(2, params.get('min_replicas', 1))
        else:
            params['spike_buffer_multiplier'] = 1.2  # Normal buffer
            
        return params
    
    def _adjust_periodic_params(self, params: Dict, uncertainty_level: str) -> Dict:
        """Adjust periodic-specific parameters based on uncertainty."""
        if uncertainty_level == 'high':
            # Use reactive scaling instead of predictive when uncertain
            params['use_predictive'] = False
            params['prediction_window'] = 0
        elif uncertainty_level == 'medium':
            # Shorter prediction window
            params['use_predictive'] = True
            params['prediction_window'] = 15  # 15 minutes
        else:
            # Full predictive capabilities
            params['use_predictive'] = True
            params['prediction_window'] = 30  # 30 minutes
            
        return params
    
    def _adjust_ramp_params(self, params: Dict, uncertainty_level: str) -> Dict:
        """Adjust ramp-specific parameters based on uncertainty."""
        if uncertainty_level == 'high':
            # More reactive to actual changes
            params['trend_sensitivity'] = 0.3  # Less sensitive
            params['projection_window'] = 5  # Short projection
        elif uncertainty_level == 'medium':
            params['trend_sensitivity'] = 0.5
            params['projection_window'] = 10
        else:
            params['trend_sensitivity'] = 0.7  # More sensitive
            params['projection_window'] = 15
            
        return params
    
    def _compute_safety_margin(self, uncertainty_level: str, archetype: str) -> float:
        """Compute safety margin for replica count based on uncertainty."""
        base_margins = {
            'SPIKE': 0.3,      # 30% extra for spikes
            'PERIODIC': 0.1,   # 10% for periodic
            'RAMP': 0.2,       # 20% for ramps
            'STATIONARY_NOISY': 0.15  # 15% for stationary
        }
        
        uncertainty_multipliers = {
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0
        }
        
        base_margin = base_margins.get(archetype, 0.2)
        multiplier = uncertainty_multipliers.get(uncertainty_level, 1.5)
        
        return base_margin * multiplier
    
    def compute_target_replicas(self,
                              current_replicas: int,
                              current_load: float,
                              predicted_load: float,
                              archetype: str,
                              confidence: float) -> int:
        """Compute target replica count with uncertainty awareness.
        
        Args:
            current_replicas: Current number of replicas
            current_load: Current system load (e.g., CPU utilization)
            predicted_load: Predicted future load
            archetype: Predicted workload archetype
            confidence: Confidence in the prediction
            
        Returns:
            Target number of replicas
        """
        # Get uncertainty-adjusted parameters
        params = self.compute_uncertainty_adjusted_params(archetype, confidence, current_load)
        
        # Base calculation
        target_utilization = params.get('target_cpu_utilization', 0.7)
        safety_margin = params.get('safety_margin', 0.2)
        
        # Calculate base target
        if predicted_load > 0:
            base_target = current_replicas * (predicted_load / target_utilization)
        else:
            base_target = current_replicas * (current_load / target_utilization)
            
        # Apply safety margin
        target_with_margin = base_target * (1 + safety_margin)
        
        # Apply archetype-specific logic
        if archetype == 'SPIKE' and 'spike_buffer_multiplier' in params:
            # For spikes, ensure we have buffer capacity
            spike_buffer = params['spike_buffer_multiplier']
            target_with_margin = max(target_with_margin, current_replicas * spike_buffer)
            
        # Round up and apply min/max constraints
        target = int(np.ceil(target_with_margin))
        min_replicas = params.get('min_replicas', 1)
        max_replicas = params.get('max_replicas', 100)
        
        return max(min_replicas, min(target, max_replicas))
    
    def get_scaling_decision_explanation(self,
                                       archetype: str,
                                       confidence: float,
                                       current_replicas: int,
                                       target_replicas: int,
                                       params: Dict) -> str:
        """Generate human-readable explanation of scaling decision."""
        uncertainty_level = params.get('uncertainty_level', 'unknown')
        safety_margin = params.get('safety_margin', 0) * 100
        
        explanation = (
            f"Scaling Decision:\n"
            f"  Predicted archetype: {archetype} (confidence: {confidence:.2f})\n"
            f"  Uncertainty level: {uncertainty_level}\n"
            f"  Current replicas: {current_replicas}\n"
            f"  Target replicas: {target_replicas}\n"
            f"  Safety margin: {safety_margin:.1f}%\n"
        )
        
        if uncertainty_level == 'high':
            explanation += "  Note: High uncertainty - using conservative scaling\n"
        elif archetype == 'SPIKE' and target_replicas > current_replicas * 1.5:
            explanation += "  Note: Spike detected - aggressive scale-up\n"
            
        return explanation


# Default scaling strategies
DEFAULT_SCALING_STRATEGIES = {
    'SPIKE': {
        'target_cpu_utilization': 0.4,  # Low target for headroom
        'scale_down_cooldown': 20,      # Long cooldown (minutes)
        'min_replicas': 2,
        'max_replicas': 100,
        'scale_up_rate': 2.0,           # Double quickly
    },
    'PERIODIC': {
        'target_cpu_utilization': 0.7,  # Higher target for efficiency
        'scale_down_cooldown': 5,       # Short cooldown
        'min_replicas': 1,
        'max_replicas': 50,
        'scale_up_rate': 1.5,
    },
    'RAMP': {
        'target_cpu_utilization': 0.6,  # Medium target
        'scale_down_cooldown': 10,      # Medium cooldown
        'min_replicas': 1,
        'max_replicas': 75,
        'scale_up_rate': 1.3,
    },
    'STATIONARY_NOISY': {
        'target_cpu_utilization': 0.6,  # Medium target
        'scale_down_cooldown': 15,      # Longer to avoid oscillation
        'min_replicas': 1,
        'max_replicas': 30,
        'scale_up_rate': 1.2,
    }
}


if __name__ == "__main__":
    # Test uncertainty-aware scaling
    print("Testing Uncertainty-Aware Scaling...")
    
    scaler = UncertaintyAwareScaler(DEFAULT_SCALING_STRATEGIES)
    
    # Test scenarios
    scenarios = [
        # (archetype, confidence, current_replicas, current_load, predicted_load)
        ('SPIKE', 0.9, 5, 0.6, 2.0),      # High confidence spike
        ('SPIKE', 0.5, 5, 0.6, 2.0),      # Low confidence spike
        ('PERIODIC', 0.85, 10, 0.7, 0.8),  # High confidence periodic
        ('PERIODIC', 0.4, 10, 0.7, 0.8),   # Low confidence periodic
        ('RAMP', 0.7, 8, 0.5, 0.9),        # Medium confidence ramp
    ]
    
    for archetype, confidence, current_replicas, current_load, predicted_load in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {archetype} with confidence {confidence}")
        
        params = scaler.compute_uncertainty_adjusted_params(archetype, confidence, current_load)
        target = scaler.compute_target_replicas(
            current_replicas, current_load, predicted_load, archetype, confidence
        )
        
        explanation = scaler.get_scaling_decision_explanation(
            archetype, confidence, current_replicas, target, params
        )
        print(explanation)
        
        print(f"Key adjusted parameters:")
        for key in ['target_cpu_utilization', 'safety_margin', 'scale_down_cooldown']:
            if key in params:
                print(f"  {key}: {params[key]}")