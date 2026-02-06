import functools
import inspect
try:
    from langfuse import observe as langfuse_observe
except ImportError:
    # Fallback or older version
    from langfuse.decorators import observe as langfuse_observe

# helper for context
try:
    from langfuse.decorators import langfuse_context
except ImportError:
    langfuse_context = None

from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings

settings = get_settings()

def get_current_trace_id():
    """Returns the current Langfuse Trace ID if available."""
    if langfuse_context:
        try:
            return langfuse_context.get_current_trace_id()
        except:
            return None
    return None

def observable(name: str = None, as_type: str = "generation"):
    """
    Wrapper around Langfuse observe decorator to enforce consistency
    and allow switching observability providers if needed.
    
    Args:
        name: Name of the trace/span. If None, function name is used.
        as_type: 'generation' (LLM call) or 'span' (Logic step)
    """
    def decorator(func):
        # We can add custom pre-processing here if needed
        # For now, we delegate to Langfuse's native decorator
        # utilizing their async accumulation capabilities.
        
        # Note: Langfuse SDK automatically handles async/sync functions
        @langfuse_observe(name=name, as_type=as_type)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        @langfuse_observe(name=name, as_type=as_type)
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
