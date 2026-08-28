# tests/test_page.py
import dashboard as d


def test_page_markers():
    h = d.HTML_PAGE
    for marker in [
        'id="builder"', 'id="builder-body"', 'id="b-family"', 'id="b-variant"',
        'id="b-ctx"', 'id="b-ctx-custom"', 'id="b-desc"', 'id="b-tags"',
        'id="b-adv"', 'id="b-adv-body"', 'id="b-temp"',
        'id="b-topp"', 'id="b-repp"', 'id="b-reason"', 'id="b-think"',
        'id="b-template"', 'id="b-kv"', 'id="b-sampling-note"', 'id="b-warn"',
        'id="b-error"', 'id="b-docs"',
        'function toggleBuilder', 'function loadFamilies', 'function onFamilyChange',
        'function onVariantChange', 'function onCtxChange', 'function renderWarn',
        'function advToggle', 'function saveCustomModel', 'function toast',
        'function delCustom', 'class="custom-badge"', 'class="del-btn"',
        "t.className = 'builder-toast'",
    ]:
        assert marker in h, marker
    assert "m.custom ? " in h


def test_log_row_colspan_unchanged():
    assert '<td colspan="11">' in d.HTML_PAGE
