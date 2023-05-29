try:
    from util.evaluation import evaluate
    from util.language import spell_correction
except ImportError:
    from graphite.util.evaluation import evaluate
    from graphite.util.language import spell_correction

__all__ = ['evaluate', 'spell_correction']
