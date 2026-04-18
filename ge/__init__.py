from .alias import alias_sample, create_alias_table

__all__ = ["alias_sample", "create_alias_table"]

try:
    from .models import *  # noqa: F401,F403
except ImportError:
    pass
