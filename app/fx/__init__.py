"""FX 包初始化：导入所有特效模块触发装饰器注册。"""

from app.fx._registry import FX_REGISTRY, build_fx_catalog, apply_fx, register_fx

# 导入所有特效模块，触发 @register_fx 装饰器
import app.fx.digital_ptz    # noqa: F401
import app.fx.color          # noqa: F401
import app.fx.geometry       # noqa: F401
import app.fx.glitch         # noqa: F401
import app.fx.overlay        # noqa: F401
import app.fx.time_domain    # noqa: F401
import app.fx.stylization    # noqa: F401
import app.fx.ai_aware       # noqa: F401
import app.fx.pink_halo      # noqa: F401

__all__ = ["FX_REGISTRY", "build_fx_catalog", "apply_fx", "register_fx"]
