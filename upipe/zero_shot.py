from exorde_data import Classification, Translation

class TooBigError(Exception):
    pass

def zero_shot(item: Translation, lab_configuration, max_depth=None, depth=0) -> Classification:
    return Classification(label="", score=float(0))
