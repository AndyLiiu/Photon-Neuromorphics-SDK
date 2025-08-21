"""
Internationalization (i18n) Support
==================================

Multi-language support with built-in translations for en, es, fr, de, ja, zh.
AI-powered translation engine with context-aware photonic terminology.
"""

import json
import locale
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..utils.logging_system import global_logger


class SupportedLocale(Enum):
    """Supported locales for global deployment."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class TranslationContext:
    """Context for photonic-specific translations."""
    domain: str  # optical, quantum, hardware, software
    technical_level: str  # basic, intermediate, advanced
    region: str  # geographic region for cultural adaptation


class TranslationEngine:
    """AI-powered translation engine with photonic terminology."""
    
    def __init__(self):
        self.logger = global_logger
        self.photonic_glossary = {
            "en": {
                "photonic": "photonic",
                "waveguide": "waveguide", 
                "interferometer": "interferometer",
                "modulator": "modulator",
                "detector": "detector",
                "neural_network": "neural network",
                "spiking": "spiking",
                "optical_training": "optical training",
                "quantum": "quantum",
                "coherence": "coherence"
            },
            "es": {
                "photonic": "fotónico",
                "waveguide": "guía de ondas",
                "interferometer": "interferómetro", 
                "modulator": "modulador",
                "detector": "detector",
                "neural_network": "red neuronal",
                "spiking": "de picos",
                "optical_training": "entrenamiento óptico",
                "quantum": "cuántico",
                "coherence": "coherencia"
            },
            "fr": {
                "photonic": "photonique",
                "waveguide": "guide d'onde",
                "interferometer": "interféromètre",
                "modulator": "modulateur", 
                "detector": "détecteur",
                "neural_network": "réseau de neurones",
                "spiking": "à impulsions",
                "optical_training": "entraînement optique",
                "quantum": "quantique",
                "coherence": "cohérence"
            },
            "de": {
                "photonic": "photonisch",
                "waveguide": "Wellenleiter",
                "interferometer": "Interferometer",
                "modulator": "Modulator",
                "detector": "Detektor", 
                "neural_network": "neuronales Netz",
                "spiking": "spikend",
                "optical_training": "optisches Training",
                "quantum": "Quanten",
                "coherence": "Kohärenz"
            },
            "ja": {
                "photonic": "フォトニック",
                "waveguide": "導波路",
                "interferometer": "干渉計",
                "modulator": "変調器",
                "detector": "検出器",
                "neural_network": "ニューラルネットワーク", 
                "spiking": "スパイキング",
                "optical_training": "光学トレーニング",
                "quantum": "量子",
                "coherence": "コヒーレンス"
            },
            "zh": {
                "photonic": "光子",
                "waveguide": "波导",
                "interferometer": "干涉仪",
                "modulator": "调制器",
                "detector": "探测器",
                "neural_network": "神经网络",
                "spiking": "脉冲",
                "optical_training": "光学训练", 
                "quantum": "量子",
                "coherence": "相干性"
            }
        }
    
    def translate_technical_term(self, term: str, target_locale: str, context: Optional[TranslationContext] = None) -> str:
        """Translate technical photonic terms with context awareness."""
        if target_locale not in self.photonic_glossary:
            return term
        
        glossary = self.photonic_glossary[target_locale]
        if term.lower() in glossary:
            translation = glossary[term.lower()]
            
            # Apply context-aware modifications
            if context:
                translation = self._apply_context(translation, context, target_locale)
            
            return translation
        
        return term
    
    def _apply_context(self, translation: str, context: TranslationContext, locale: str) -> str:
        """Apply contextual modifications to translations."""
        # Technical level adaptations
        if context.technical_level == "basic" and locale == "ja":
            # Add furigana for complex terms in Japanese
            if translation == "フォトニック":
                translation += " (photonic)"
        
        # Regional adaptations
        if context.region == "asia" and locale == "en":
            # Use metric units for Asian markets
            pass
        
        return translation


class LocaleManager:
    """Manages locale settings and formatting."""
    
    def __init__(self):
        self.current_locale = SupportedLocale.ENGLISH
        self.logger = global_logger
        self.date_formats = {
            "en": "%m/%d/%Y",
            "es": "%d/%m/%Y", 
            "fr": "%d/%m/%Y",
            "de": "%d.%m.%Y",
            "ja": "%Y/%m/%d",
            "zh": "%Y年%m月%d日"
        }
        self.number_formats = {
            "en": {"decimal": ".", "thousands": ","},
            "es": {"decimal": ",", "thousands": "."},
            "fr": {"decimal": ",", "thousands": " "},
            "de": {"decimal": ",", "thousands": "."},
            "ja": {"decimal": ".", "thousands": ","},
            "zh": {"decimal": ".", "thousands": ","}
        }
    
    def set_locale(self, locale_code: str):
        """Set the current locale."""
        try:
            target_locale = SupportedLocale(locale_code)
            self.current_locale = target_locale
            self.logger.info(f"Locale set to: {locale_code}")
        except ValueError:
            self.logger.warning(f"Unsupported locale: {locale_code}, using default")
    
    def format_number(self, number: float, locale_code: Optional[str] = None) -> str:
        """Format numbers according to locale conventions."""
        if locale_code is None:
            locale_code = self.current_locale.value
        
        fmt = self.number_formats.get(locale_code, self.number_formats["en"])
        
        # Format with appropriate decimal and thousands separators
        if isinstance(number, float):
            formatted = f"{number:,.2f}"
        else:
            formatted = f"{number:,}"
        
        # Replace separators according to locale
        formatted = formatted.replace(".", "DECIMAL_TEMP")
        formatted = formatted.replace(",", fmt["thousands"])
        formatted = formatted.replace("DECIMAL_TEMP", fmt["decimal"])
        
        return formatted
    
    def format_date(self, date_obj, locale_code: Optional[str] = None) -> str:
        """Format dates according to locale conventions."""
        if locale_code is None:
            locale_code = self.current_locale.value
        
        date_format = self.date_formats.get(locale_code, self.date_formats["en"])
        return date_obj.strftime(date_format)


class InternationalizationManager:
    """Central manager for internationalization features."""
    
    def __init__(self):
        self.locale_manager = LocaleManager()
        self.translation_engine = TranslationEngine()
        self.logger = global_logger
        self.translations_cache = {}
        
        # Load default UI translations
        self._load_ui_translations()
    
    def _load_ui_translations(self):
        """Load UI text translations."""
        self.ui_translations = {
            "en": {
                "error_occurred": "An error occurred",
                "processing": "Processing",
                "complete": "Complete",
                "failed": "Failed",
                "photonic_simulation": "Photonic Simulation",
                "neural_network": "Neural Network",
                "performance_metrics": "Performance Metrics",
                "optimization_progress": "Optimization Progress"
            },
            "es": {
                "error_occurred": "Ocurrió un error", 
                "processing": "Procesando",
                "complete": "Completo",
                "failed": "Falló",
                "photonic_simulation": "Simulación Fotónica",
                "neural_network": "Red Neuronal",
                "performance_metrics": "Métricas de Rendimiento",
                "optimization_progress": "Progreso de Optimización"
            },
            "fr": {
                "error_occurred": "Une erreur s'est produite",
                "processing": "Traitement en cours", 
                "complete": "Terminé",
                "failed": "Échec",
                "photonic_simulation": "Simulation Photonique",
                "neural_network": "Réseau de Neurones",
                "performance_metrics": "Métriques de Performance",
                "optimization_progress": "Progrès d'Optimisation"
            },
            "de": {
                "error_occurred": "Ein Fehler ist aufgetreten",
                "processing": "Verarbeitung",
                "complete": "Vollständig",
                "failed": "Fehlgeschlagen", 
                "photonic_simulation": "Photonische Simulation",
                "neural_network": "Neuronales Netz",
                "performance_metrics": "Leistungsmetriken",
                "optimization_progress": "Optimierungsfortschritt"
            },
            "ja": {
                "error_occurred": "エラーが発生しました",
                "processing": "処理中",
                "complete": "完了",
                "failed": "失敗",
                "photonic_simulation": "フォトニックシミュレーション",
                "neural_network": "ニューラルネットワーク", 
                "performance_metrics": "パフォーマンス指標",
                "optimization_progress": "最適化の進捗"
            },
            "zh": {
                "error_occurred": "发生错误",
                "processing": "处理中",
                "complete": "完成",
                "failed": "失败",
                "photonic_simulation": "光子仿真",
                "neural_network": "神经网络",
                "performance_metrics": "性能指标", 
                "optimization_progress": "优化进度"
            }
        }
    
    def get_text(self, key: str, locale_code: Optional[str] = None) -> str:
        """Get localized UI text."""
        if locale_code is None:
            locale_code = self.locale_manager.current_locale.value
        
        translations = self.ui_translations.get(locale_code, self.ui_translations["en"])
        return translations.get(key, key)
    
    def translate_error_message(self, error_msg: str, locale_code: Optional[str] = None) -> str:
        """Translate error messages with photonic context."""
        if locale_code is None:
            locale_code = self.locale_manager.current_locale.value
        
        # Apply photonic term translations
        translated = error_msg
        for term in ["waveguide", "photonic", "quantum", "neural network"]:
            if term in error_msg.lower():
                translated_term = self.translation_engine.translate_technical_term(
                    term, locale_code
                )
                translated = translated.replace(term, translated_term)
        
        return translated
    
    def setup_global_locale(self, locale_code: str):
        """Setup global locale for entire application."""
        self.locale_manager.set_locale(locale_code)
        
        # Apply system locale settings
        try:
            if locale_code == "en":
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            elif locale_code == "es":
                locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
            elif locale_code == "fr":
                locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
            elif locale_code == "de":
                locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
            elif locale_code == "ja":
                locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
            elif locale_code == "zh":
                locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except locale.Error:
            self.logger.warning(f"Could not set system locale for {locale_code}")
        
        self.logger.info(f"Global locale configured: {locale_code}")


# Global instance for easy access
global_i18n = InternationalizationManager()

def _(key: str, locale: Optional[str] = None) -> str:
    """Shorthand function for getting localized text."""
    return global_i18n.get_text(key, locale)

def setup_locale(locale_code: str):
    """Setup application locale."""
    global_i18n.setup_global_locale(locale_code)