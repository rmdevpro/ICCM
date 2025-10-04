# Fiedler Security Feature Review Request

**Date:** 2025-10-02
**Review Type:** Feature Addition Review
**Correlation ID:** FIEDLER-SECURITY-R1
**Reviewers:** Fiedler's default models 

## Review Scope

We added encrypted API key storage to Fiedler after the initial implementation. Please review the security feature for:

1. **Security Correctness**: Is the keyring integration secure?
2. **Implementation Quality**: Clean code, proper error handling?
3. **Backward Compatibility**: Does it break existing functionality?
4. **Documentation**: Clear security guidance?
5. **Potential Issues**: Any security vulnerabilities or bugs?

## What Was Added

### New MCP Tools (3)

1. **fiedler_set_key** - Store API key in encrypted OS keyring
2. **fiedler_list_keys** - List providers with stored keys
3. **fiedler_delete_key** - Remove stored key

### Security Architecture

**Key Retrieval Order:**
1. System keyring (encrypted) - First priority
2. Environment variable (plaintext) - Fallback
3. Error if neither found

**OS-Native Encryption:**
- macOS: Keychain (AES-256)
- Windows: Credential Manager (DPAPI)
- Linux: Secret Service (D-Bus)

### Files Changed

**New Files:**
- `fiedler/utils/secrets.py` (107 lines)
- `fiedler/tools/keys.py` (115 lines)
- `SECURITY.md` (documentation)
- `SECURITY_FEATURE_ADDED.md` (implementation notes)

**Modified Files:**
- `pyproject.toml` - Added keyring dependency
- All 4 providers updated to use `get_api_key()` with keyring fallback
- `fiedler/server.py` - Registered 3 new tools
- `README.md` - Security section added

## Review Questions

1. **Security**: Are there any vulnerabilities in the keyring integration?
2. **Fallback Logic**: Is the keyring→env var fallback safe and correct?
3. **Error Handling**: Are keyring errors handled gracefully?
4. **Key Validation**: Should we validate API key format before storage?
5. **Multi-User**: Any issues with multiple users on same system?
6. **Documentation**: Is security guidance clear for users?
7. **Backward Compatibility**: Any breaking changes?
8. **Dependencies**: Is keyring library dependency acceptable?

## Expected Review Format

Please provide:

1. **Security Assessment** (1-2 paragraphs)
2. **Issues Found** (by severity: Critical, Major, Minor)
3. **Specific Code Corrections** (with file:line references)
4. **Security Recommendations**
5. **Testing Recommendations**

---

# New/Modified Code

## fiedler/utils/secrets.py (NEW)

```python
"""Secure API key management using system keyring."""
import os
from typing import Optional

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False


SERVICE_NAME = "fiedler-mcp-server"


def get_api_key(provider: str, env_var_name: str) -> Optional[str]:
    """
    Get API key from keyring or environment variable.

    Fallback order:
    1. System keyring (secure, encrypted)
    2. Environment variable
    3. None (caller should handle missing key)

    Args:
        provider: Provider name (e.g., "google", "openai", "together", "xai")
        env_var_name: Environment variable name to check as fallback

    Returns:
        API key string or None
    """
    # Try keyring first (most secure)
    if KEYRING_AVAILABLE:
        try:
            key = keyring.get_password(SERVICE_NAME, provider)
            if key:
                return key
        except Exception:
            # Keyring errors shouldn't break normal operation
            pass

    # Fall back to environment variable
    return os.getenv(env_var_name)


def set_api_key(provider: str, api_key: str) -> None:
    """
    Store API key in system keyring.

    Args:
        provider: Provider name (e.g., "google", "openai", "together", "xai")
        api_key: API key to store (will be encrypted by OS keyring)

    Raises:
        RuntimeError: If keyring is not available
    """
    if not KEYRING_AVAILABLE:
        raise RuntimeError(
            "Keyring library not available. Install with: pip install keyring"
        )

    keyring.set_password(SERVICE_NAME, provider, api_key)


def delete_api_key(provider: str) -> bool:
    """
    Delete API key from system keyring.

    Args:
        provider: Provider name

    Returns:
        True if key was deleted, False if no key was stored

    Raises:
        RuntimeError: If keyring is not available
    """
    if not KEYRING_AVAILABLE:
        raise RuntimeError(
            "Keyring library not available. Install with: pip install keyring"
        )

    try:
        existing = keyring.get_password(SERVICE_NAME, provider)
        if existing:
            keyring.delete_password(SERVICE_NAME, provider)
            return True
        return False
    except keyring.errors.PasswordDeleteError:
        return False


def list_stored_providers() -> list[str]:
    """
    List providers that have keys stored in keyring.

    Returns:
        List of provider names with stored keys
    """
    if not KEYRING_AVAILABLE:
        return []

    # Known providers from config
    known_providers = ["google", "openai", "together", "xai"]

    stored = []
    for provider in known_providers:
        try:
            if keyring.get_password(SERVICE_NAME, provider):
                stored.append(provider)
        except Exception:
            continue

    return stored
```

## fiedler/tools/keys.py (NEW)

```python
"""API key management tools."""
from typing import Dict, Any

from ..utils.secrets import set_api_key, delete_api_key, list_stored_providers, KEYRING_AVAILABLE


def fiedler_set_key(provider: str, api_key: str) -> Dict[str, Any]:
    """
    Store API key securely in system keyring.

    Args:
        provider: Provider name (google, openai, together, xai)
        api_key: API key to store (encrypted by OS keyring)

    Returns:
        Dict with status and message
    """
    # Validate provider
    valid_providers = ["google", "openai", "together", "xai"]
    if provider not in valid_providers:
        raise ValueError(
            f"Invalid provider '{provider}'. Must be one of: {', '.join(valid_providers)}"
        )

    # Validate key is not empty
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty")

    # Store in keyring
    try:
        set_api_key(provider, api_key)
        return {
            "status": "success",
            "provider": provider,
            "message": f"API key stored securely for {provider}",
            "storage": "system_keyring"
        }
    except RuntimeError as e:
        # Keyring not available
        return {
            "status": "error",
            "provider": provider,
            "message": str(e),
            "storage": "unavailable"
        }


def fiedler_delete_key(provider: str) -> Dict[str, Any]:
    """
    Delete stored API key from system keyring.

    Args:
        provider: Provider name (google, openai, together, xai)

    Returns:
        Dict with status and message
    """
    # Validate provider
    valid_providers = ["google", "openai", "together", "xai"]
    if provider not in valid_providers:
        raise ValueError(
            f"Invalid provider '{provider}'. Must be one of: {', '.join(valid_providers)}"
        )

    try:
        deleted = delete_api_key(provider)
        if deleted:
            return {
                "status": "success",
                "provider": provider,
                "message": f"API key deleted for {provider}"
            }
        else:
            return {
                "status": "not_found",
                "provider": provider,
                "message": f"No API key stored for {provider}"
            }
    except RuntimeError as e:
        return {
            "status": "error",
            "provider": provider,
            "message": str(e)
        }


def fiedler_list_keys() -> Dict[str, Any]:
    """
    List providers with stored API keys.

    Returns:
        Dict with list of providers and keyring availability
    """
    stored = list_stored_providers()

    return {
        "keyring_available": KEYRING_AVAILABLE,
        "providers_with_keys": stored,
        "count": len(stored),
        "message": f"{len(stored)} provider(s) have stored keys" if KEYRING_AVAILABLE
                   else "Keyring not available - install with: pip install keyring"
    }
```

## Provider Updates (Example: fiedler/providers/openai.py)

```python
# Before
def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str):
    super().__init__(model_id, config)
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable: {api_key_env}")
    self.client = OpenAI(api_key=api_key)

# After
from ..utils.secrets import get_api_key

def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str):
    super().__init__(model_id, config)
    # Check keyring first, then env var
    api_key = get_api_key("openai", api_key_env)
    if not api_key:
        raise ValueError(
            f"No API key found for OpenAI. Set via fiedler_set_key or environment variable {api_key_env}"
        )
    self.client = OpenAI(api_key=api_key)
```

**All 4 providers (gemini, openai, together, xai) updated with same pattern.**

## MCP Server Registration (fiedler/server.py)

```python
# Added to list_tools():
Tool(
    name="fiedler_set_key",
    description="Store API key securely in system keyring (encrypted). Replaces need for environment variables.",
    inputSchema={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider name: google, openai, together, or xai",
            },
            "api_key": {
                "type": "string",
                "description": "API key to store (will be encrypted by OS keyring)",
            },
        },
        "required": ["provider", "api_key"],
    },
),
Tool(
    name="fiedler_delete_key",
    description="Delete stored API key from system keyring.",
    inputSchema={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider name: google, openai, together, or xai",
            },
        },
        "required": ["provider"],
    },
),
Tool(
    name="fiedler_list_keys",
    description="List providers that have API keys stored in system keyring.",
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
),

# Added to call_tool():
elif name == "fiedler_set_key":
    result = fiedler_set_key(
        provider=arguments["provider"],
        api_key=arguments["api_key"],
    )
elif name == "fiedler_delete_key":
    result = fiedler_delete_key(provider=arguments["provider"])
elif name == "fiedler_list_keys":
    result = fiedler_list_keys()
```

## Dependency Addition (pyproject.toml)

```toml
dependencies = [
    "pyyaml>=6.0.1",
    "openai>=1.37.0",
    "mcp>=1.2.0",
    "keyring>=24.0.0",  # NEW: OS-native keyring integration
]
```

---

# End of Security Feature Addition

Please provide your detailed security review following the format specified above.
