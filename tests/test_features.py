from fake_news.utils.features import compute_bin_idx
from fake_news.utils.features import normalize_and_clean_counts
from fake_news.utils.features import normalize_and_clean_party_affiliations
from fake_news.utils.features import normalize_and_clean_speaker_title
from fake_news.utils.features import normalize_and_clean_state_info
from fake_news.utils.features import normalize_labels


def test_compute_bin_idx():
    bins = [0, 4, 10, 12]
    assert compute_bin_idx(0, bins) == 0
    assert compute_bin_idx(3, bins) == 1
    assert compute_bin_idx(4, bins) == 1
    assert compute_bin_idx(12, bins) == 3


def test_normalize_labels():
    datapoints = [
        {"label": "pants-fire", "ignored_field": "blah"},
        {"label": "barely-true"},
        {"label": "false"},
        {"label": "true"},
        {"label": "half-true"},
        {"label": "mostly-true"}
    ]
    
    expected_converted_datapoints = [
        {"label": False, "ignored_field": "blah"},
        {"label": False},
        {"label": False},
        {"label": True},
        {"label": True},
        {"label": True}
    ]
    
    assert normalize_labels(datapoints) == expected_converted_datapoints


def test_normalize_speaker_title():
    datapoints = [
        {"speaker_title": "mr-president ", "ignored_label": "true"},
        {"speaker_title": "  U. S. CONGRESSMAN"}
    ]
    
    expected_converted_datapoints = [
        {"speaker_title": "mr president", "ignored_label": "true"},
        {"speaker_title": "u.s. congressman"}
    ]
    
    assert normalize_and_clean_speaker_title(datapoints) == expected_converted_datapoints


def test_normalize_party_affiliations():
    datapoints = [
        {"party_affiliation": "democrat", "ignored_label": "true"},
        {"party_affiliation": "boston tea"}
    ]
    
    expected_converted_datapoints = [
        {"party_affiliation": "democrat", "ignored_label": "true"},
        {"party_affiliation": "none"}
    ]
    
    assert normalize_and_clean_party_affiliations(datapoints) == expected_converted_datapoints


def test_normalize_state_info():
    datapoints = [
        {"state_info": " Virgina ", "ignored_label": "true"},
        {"state_info": " TEX "}
    ]
    
    expected_converted_datapoints = [
        {"state_info": "virginia", "ignored_label": "true"},
        {"state_info": "texas"}
    ]
    
    assert normalize_and_clean_state_info(datapoints) == expected_converted_datapoints


def test_normalize_counts():
    datapoints = [
        {"barely_true_count": "23.0", "ignored_label": "true"},
        {"false_count": "1.0"}
    ]
    
    expected_converted_datapoints = [
        {"barely_true_count": 23, "ignored_label": "true"},
        {"false_count": 1}
    ]
    
    assert normalize_and_clean_counts(datapoints) == expected_converted_datapoints
