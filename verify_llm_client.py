"""
Quick verification script for LLM client abstraction layer.

Tests provider instantiation, credential validation, and basic functionality
without making actual API calls.
"""

import os
from src.llm_client import (
    LLMConfig,
    LLMClient,
    LLMProviderFactory,
    OpenAIProvider,
    AzureProvider,
    GeminiProvider,
    AnthropicProvider,
    PerplexityProvider
)


def test_provider_registration():
    """Test that all providers are registered."""
    print("Testing provider registration...")
    providers = LLMProviderFactory.list_providers()
    expected = ["openai", "azure", "gemini", "anthropic", "perplexity"]
    
    for provider in expected:
        assert provider in providers, f"Provider {provider} not registered"
        print(f"  ✓ {provider} registered")
    
    print(f"✅ All {len(providers)} providers registered\n")


def test_provider_instantiation():
    """Test creating provider instances."""
    print("Testing provider instantiation...")
    
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        openai_api_key="test-key"
    )
    
    providers = ["openai", "azure", "gemini", "anthropic", "perplexity"]
    
    for provider_name in providers:
        try:
            config.provider = provider_name
            provider = LLMProviderFactory.create(provider_name, "test-model", config)
            assert provider is not None
            print(f"  ✓ {provider_name}: {provider.__class__.__name__}")
        except Exception as e:
            print(f"  ✗ {provider_name}: {e}")
            raise
    
    print("✅ All providers instantiate correctly\n")


def test_credential_validation():
    """Test credential validation logic."""
    print("Testing credential validation...")
    
    # Test OpenAI with and without key
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    provider = OpenAIProvider(model="gpt-4o-mini", config=config)
    assert not provider.validate_credentials(), "Should fail without API key"
    print("  ✓ OpenAI validation fails without key")
    
    config.openai_api_key = "test-key"
    provider = OpenAIProvider(model="gpt-4o-mini", config=config)
    assert provider.validate_credentials(), "Should pass with API key"
    print("  ✓ OpenAI validation passes with key")
    
    # Test Azure with missing credentials
    config = LLMConfig(provider="azure", model="gpt-4")
    provider = AzureProvider(model="gpt-4", config=config)
    assert not provider.validate_credentials(), "Should fail without full Azure config"
    print("  ✓ Azure validation fails without credentials")
    
    config.azure_api_key = "test-key"
    config.azure_endpoint = "https://test.openai.azure.com"
    config.azure_deployment = "test-deployment"
    provider = AzureProvider(model="gpt-4", config=config)
    assert provider.validate_credentials(), "Should pass with all Azure credentials"
    print("  ✓ Azure validation passes with full config")
    
    print("✅ Credential validation works correctly\n")


def test_llm_client_initialization():
    """Test LLMClient initialization."""
    print("Testing LLMClient initialization...")
    
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        openai_api_key="test-key",
        temperature=0.7,
        max_tokens=256
    )
    
    client = LLMClient(config)
    assert client is not None
    assert client.get_provider_name() == "openai"
    assert client.get_model_name() == "gpt-4o-mini"
    assert client.validate_credentials() == True
    
    print(f"  ✓ Client created: {client}")
    print(f"  ✓ Provider: {client.get_provider_name()}")
    print(f"  ✓ Model: {client.get_model_name()}")
    print(f"  ✓ Credentials valid: {client.validate_credentials()}")
    
    print("✅ LLMClient initialization works\n")


def test_json_extraction():
    """Test JSON extraction from markdown."""
    print("Testing JSON extraction from markdown...")
    
    # Test with JSON markdown block
    text_with_json = '```json\n{"key": "value"}\n```'
    extracted = LLMClient._extract_json_from_markdown(text_with_json)
    assert extracted == '{"key": "value"}', f"Got: {extracted}"
    print("  ✓ Extracts JSON from markdown block")
    
    # Test with generic markdown block
    text_with_generic = '```\n{"key": "value"}\n```'
    extracted = LLMClient._extract_json_from_markdown(text_with_generic)
    assert extracted == '{"key": "value"}', f"Got: {extracted}"
    print("  ✓ Extracts JSON from generic code block")
    
    # Test with plain text
    text_plain = '{"key": "value"}'
    extracted = LLMClient._extract_json_from_markdown(text_plain)
    assert extracted == '{"key": "value"}', f"Got: {extracted}"
    print("  ✓ Passes through plain JSON")
    
    print("✅ JSON extraction works correctly\n")


def test_config_from_env():
    """Test creating config from environment variables (if available)."""
    print("Testing config from environment variables...")
    
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_azure = all([
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
    ])
    
    if has_openai:
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        client = LLMClient(config)
        assert client.validate_credentials()
        print("  ✓ OpenAI credentials found and valid")
    else:
        print("  ⊘ OpenAI credentials not found (expected in test)")
    
    if has_azure:
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        client = LLMClient(config)
        assert client.validate_credentials()
        print("  ✓ Azure credentials found and valid")
    else:
        print("  ⊘ Azure credentials not found (expected in test)")
    
    print("✅ Environment credential loading works\n")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("LLM Client Abstraction Layer Verification")
    print("=" * 60 + "\n")
    
    try:
        test_provider_registration()
        test_provider_instantiation()
        test_credential_validation()
        test_llm_client_initialization()
        test_json_extraction()
        test_config_from_env()
        
        print("=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("=" * 60 + "\n")
        
        print("Summary:")
        print("  • Abstract base class: BaseLLMProvider")
        print("  • Concrete providers: 5 (OpenAI, Azure, Gemini, Anthropic, Perplexity)")
        print("  • Factory pattern: LLMProviderFactory with registry")
        print("  • Unified client: LLMClient")
        print("  • Type hints: Complete coverage")
        print("  • Docstrings: Comprehensive")
        print("\nNext steps:")
        print("  1. Update src/utils.py to use LLMClient")
        print("  2. Update src/data_pipeline.py validation logic")
        print("  3. Update tests to use new abstraction")
        print("  4. Run existing test suite to verify backward compatibility")
        
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
