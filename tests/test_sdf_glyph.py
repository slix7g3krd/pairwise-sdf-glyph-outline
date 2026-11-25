import pytest
import graphics.sdf_glyph as mod

CASES = [
    ("atlas_upload_fence", (1,)),
    ("screen_space_outline", (1,)),
    ("atlas_ref_count", (1,)),
    ("layout_snapshot", (1,)),
    ("dpi_relayout", (1,)),
    ("fallback_baseline_unify", (1,)),
    ("load_font", (1,)),
    ("layout_text_sdf", (1,)),
    ("raster_glyph", (1,)),
    ("present_sdf_text", (1,)),
    ("validate_sdf_text", (1,)),
    ("metrics_sdf_text", (1,)),
    ("audit_sdf_text", (1,)),
    ("fallback_sdf_text", (1,)),
    ("determinism_sdf_text", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
