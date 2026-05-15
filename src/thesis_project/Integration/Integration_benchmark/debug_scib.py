.# src/thesis_project/benchmark/debug_scib.py
import inspect
import importlib

def debug_kbet_signature():
    print("=== scIB kBET signature debug ===")
    try:
        scib = importlib.import_module("scib")
        print("✓ scib imported:", getattr(scib, "__version__", "unknown"))

        # Try scib.metrics.kbet.kBET first
        try:
            mod = importlib.import_module("scib.metrics.kbet")
            fn = getattr(mod, "kBET", None) or getattr(mod, "kbet", None)
            if fn is not None:
                print("✓ Found:", f"{mod.__name__}.{fn.__name__}")
                print("Signature:", inspect.signature(fn))
                print("Parameters:", list(inspect.signature(fn).parameters.keys()))
                return
        except Exception as e:
            print("scib.metrics.kbet not usable:", type(e).__name__, e)

        # Try scib.metrics.kBET
        try:
            mod = importlib.import_module("scib.metrics")
            fn = getattr(mod, "kBET", None) or getattr(mod, "kbet", None)
            if fn is not None:
                print("✓ Found:", f"{mod.__name__}.{fn.__name__}")
                print("Signature:", inspect.signature(fn))
                print("Parameters:", list(inspect.signature(fn).parameters.keys()))
                return
        except Exception as e:
            print("scib.metrics not usable:", type(e).__name__, e)

        # Try scib.kBET
        fn = getattr(scib, "kBET", None) or getattr(scib, "kbet", None)
        if fn is not None:
            print("✓ Found:", f"scib.{fn.__name__}")
            print("Signature:", inspect.signature(fn))
            print("Parameters:", list(inspect.signature(fn).parameters.keys()))
            return

        print("✗ Could not find kBET callable in scib.")
    except Exception as e:
        print("✗ scib not importable:", type(e).__name__, e)
