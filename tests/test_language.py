from tnmi.language import detect_language


def test_detects_tamil():
    assert detect_language("தமிழக அரசு இன்று புதிய திட்டம் அறிவித்தது") == "ta"


def test_detects_english():
    assert detect_language("The Tamil Nadu government announced a new scheme") == "en"


def test_detects_mixed_tamil_english():
    assert detect_language("தமிழக அரசு new scheme announce செய்தது") == "ta-en-mixed"


def test_detects_unknown_for_empty_text():
    assert detect_language("   ") == "unknown"
