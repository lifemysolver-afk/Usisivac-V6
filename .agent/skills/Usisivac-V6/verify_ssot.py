from dynaconf import Dynaconf
from loguru import logger
from cerberus import Validator
import sys
import os

# 1. Konfiguracija Loguru-a za strukturirano logovanje
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("WORK_LOG.json", serialize=True, rotation="10 MB")

# 2. Definisanje šeme za validaciju (Cerberus)
CONFIG_SCHEMA = {
    'cpu_timeout_seconds': {'type': 'integer', 'min': 1, 'max': 3600},
    'validation_thresholds': {
        'type': 'dict',
        'schema': {
            'baseline_oof_score': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'oof_calibration_rule': {
                'type': 'dict',
                'schema': {
                    'threshold': {'type': 'float', 'min': 0.0, 'max': 1.0}
                }
            }
        }
    }
}

def verify_ssot_parameters():
    """
    Učitava, validira i loguje SSOT parametre koristeći Dynaconf, Cerberus i Loguru.
    """
    try:
        # 3. Učitavanje konfiguracije (Dynaconf)
        settings = Dynaconf(
            settings_files=['settings.toml', '.secrets.toml'],
            environments=True,
            load_dotenv=True,
        )

        # 4. Validacija (Cerberus)
        v = Validator(CONFIG_SCHEMA)
        config_dict = settings.to_dict()

        if not v.validate(config_dict):
            logger.error(f"Validacija konfiguracije neuspešna: {v.errors}")
            return None

        # 5. Strukturirano logovanje uspeha
        logger.info("SSOT parametri uspešno učitani i validirani",
                    extra={
                        "cpu_timeout": settings.cpu_timeout_seconds,
                        "baseline_score": settings.validation_thresholds.baseline_oof_score,
                        "calibration_threshold": settings.validation_thresholds.oof_calibration_rule.threshold
                    })

        return settings

    except Exception as e:
        logger.exception(f"Neočekivana greška pri verifikaciji SSOT-a: {e}")
        return None

if __name__ == "__main__":
    logger.info("Pokretanje SSOT verifikacije...")
    ssot_settings = verify_ssot_parameters()

    if ssot_settings:
        print("\n✅ SSOT Verifikacija Uspešna!")
        print(f"  CPU Timeout: {ssot_settings.cpu_timeout_seconds}s")
        print(f"  Baseline OOF: {ssot_settings.validation_thresholds.baseline_oof_score}")
    else:
        print("\n❌ SSOT Verifikacija Neuspešna! Proverite logove.")
        sys.exit(1)
