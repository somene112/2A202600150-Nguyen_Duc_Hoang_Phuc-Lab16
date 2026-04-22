from src.reflexion_lab.utils import normalize_answer

def test_normalize_answer():
    assert normalize_answer("Oxford University!") == "oxford university"
