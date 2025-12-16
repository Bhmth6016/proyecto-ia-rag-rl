#!/usr/bin/env python3
"""Verificar configuración ML y NLP"""

from src.core.config import settings

print("🔍 VERIFICANDO CONFIGURACIÓN ML/NLP")
print("="*50)

# Probar diferentes modos
for mode in ["basic", "balanced", "enhanced"]:
    print(f"\n🧪 Modo: {mode}")
    settings.apply_mode_config(mode)
    print(f"   • ML_ENABLED: {settings.ML_ENABLED}")
    print(f"   • NLP_ENABLED: {settings.NLP_ENABLED}")
    print(f"   • ML_FEATURES: {list(settings.ML_FEATURES)}")

# Test específico de modo enhanced
print(f"\n🔥 Forzando modo enhanced...")
settings.apply_mode_config("enhanced")
print(f"   • ML_ENABLED: {settings.ML_ENABLED}")
print(f"   • NLP_ENABLED: {settings.NLP_ENABLED}")
print(f"   • Tiene NER: {'ner' in settings.ML_FEATURES}")
print(f"   • Tiene Zero-Shot: {'zero_shot' in settings.ML_FEATURES}")

# Test de importación NLP
try:
    from src.core.nlp.enrichment import NLPEnricher
    print("✅ NLPEnricher disponible")
    
    # Test rápido
    nlp = NLPEnricher(device="cpu", use_small_models=True)
    nlp.initialize()
    
    test_text = "Quiero un laptop gaming ASUS con 16GB RAM"
    entities = nlp.extract_entities(test_text)
    print(f"🔍 Entidades en '{test_text}':")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"   • {entity_type}: {entity_list}")
    
    nlp.cleanup_memory()
    print("✅ Test NLP completado")
    
except ImportError as e:
    print(f"❌ NLPEnricher no disponible: {e}")
except Exception as e:
    print(f"⚠️  Error en test NLP: {e}")

print("\n✅ Verificación completada")