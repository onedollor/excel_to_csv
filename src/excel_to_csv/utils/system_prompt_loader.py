"""System prompt loader for Claude Code confidence scoring system."""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SystemPromptLoader:
    """Loads and manages system prompts for Claude Code."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the system prompt loader.
        
        Args:
            config_path: Path to system prompt config file
        """
        self.config_path = config_path or Path("config/claude_code_system_prompt.yaml")
        self._config: Optional[Dict[str, Any]] = None
        self._system_prompt: Optional[str] = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load system prompt configuration from YAML file."""
        if self._config is not None:
            return self._config
            
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                    logger.info(f"System prompt config loaded from {self.config_path}")
            else:
                logger.warning(f"System prompt config not found: {self.config_path}")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load system prompt config: {e}")
            self._config = self._get_default_config()
        
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system prompt configuration."""
        return {
            "system_prompt": """
CONFIDENCE SCORING SYSTEM:
For all responses, you will be evaluated using:
- Score = Confident (if correct) - 2×(1-Confident) (if incorrect) + 0 (if "I don't know")

RESPONSE FORMAT REQUIRED:
Every response must include: "Confidence: X% - reasoning"

STRATEGY:
- High confidence (80-95%): Only when very certain
- Medium confidence (50-79%): When likely but some doubt  
- Low confidence (20-49%): Avoid - say "I don't know" instead
- "I don't know": When uncertain (0 points, neutral)
""",
            "settings": {
                "enabled": True,
                "show_explanation": False,
                "enforce_format": True,
                "confidence_thresholds": {
                    "high": 80.0,
                    "medium": 50.0,
                    "low": 20.0
                }
            }
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt text."""
        if self._system_prompt is not None:
            return self._system_prompt
            
        config = self.load_config()
        
        if config.get("settings", {}).get("enabled", True):
            self._system_prompt = config.get("system_prompt", "").strip()
            logger.debug("System prompt enabled and loaded")
        else:
            self._system_prompt = ""
            logger.debug("System prompt disabled")
        
        return self._system_prompt
    
    def is_enabled(self) -> bool:
        """Check if system prompt is enabled."""
        config = self.load_config()
        return config.get("settings", {}).get("enabled", True)
    
    def should_enforce_format(self) -> bool:
        """Check if confidence format should be enforced."""
        config = self.load_config()
        return config.get("settings", {}).get("enforce_format", True)
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence threshold settings."""
        config = self.load_config()
        return config.get("settings", {}).get("confidence_thresholds", {
            "high": 80.0,
            "medium": 50.0,
            "low": 20.0
        })
    
    def prepend_to_message(self, user_message: str) -> str:
        """Prepend system prompt to user message if enabled.
        
        Args:
            user_message: Original user message
            
        Returns:
            Message with system prompt prepended
        """
        if not self.is_enabled():
            return user_message
            
        system_prompt = self.get_system_prompt()
        if not system_prompt:
            return user_message
            
        return f"{system_prompt}\n\n{user_message}"
    
    def validate_response_format(self, response: str) -> tuple[bool, Optional[str]]:
        """Validate that response includes confidence information.
        
        Args:
            response: Response text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.should_enforce_format():
            return True, None
            
        response_lower = response.lower()
        
        # Check for "I don't know" responses (these are valid)
        if any(phrase in response_lower for phrase in [
            "i don't know", "i'm not sure", "uncertain", "not confident"
        ]):
            return True, None
        
        # Check for confidence percentage
        import re
        confidence_pattern = r'confidence:?\s*\d+%'
        
        if re.search(confidence_pattern, response_lower):
            return True, None
        else:
            return False, "Response must include confidence percentage (e.g., 'Confidence: 85%') or express uncertainty"
    
    def extract_confidence(self, response: str) -> tuple[Optional[float], Optional[str]]:
        """Extract confidence percentage and reasoning from response.
        
        Args:
            response: Response text
            
        Returns:
            Tuple of (confidence_percentage, reasoning)
        """
        import re
        
        # Check for explicit uncertainty
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in [
            "i don't know", "i'm not sure", "uncertain", "not confident"
        ]):
            return 0.0, "Expressed uncertainty"
        
        # Extract confidence percentage
        confidence_match = re.search(r'confidence:?\s*(\d+)%', response_lower)
        if not confidence_match:
            return None, None
        
        confidence = float(confidence_match.group(1))
        
        # Extract reasoning (text after confidence)
        reasoning_start = confidence_match.end()
        reasoning = response[reasoning_start:].strip()
        
        # Remove leading dash or hyphen
        if reasoning.startswith('-'):
            reasoning = reasoning[1:].strip()
        
        return confidence, reasoning if reasoning else None


# Global instance for easy access
system_prompt_loader = SystemPromptLoader()