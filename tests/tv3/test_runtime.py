"""TV3 contract tests. Synthetic inputs validate code, not fruit accuracy.

This version selects Streamlit controls by their meaning instead of their
index, so the Nhi UI can add a clear-image button without breaking TV3 tests.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.data import rgb_from_bytes
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
from freshlens_ai.inference.open_set import analyze_bytes
from freshlens_ai.inference.quality import QualityConfig
from freshlens_ai.inference.gradcam import GradCAM
from freshlens_ai.evaluation.tv3 import summarize


@pytest.fixture(scope='module')
def runtime():
    torch.set_num_threads(2)
    return FreshLensPredictor(PROJECT_DIR / 'models/cnn_efficientnet_b0/best.pt', device='cpu')


def encoded(image, fmt='PNG', **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize('kind', ['wide', 'alpha', 'exif', 'gray', 'webp'])
def test_matches_production_preprocessing_and_gate(runtime, kind):
    image = Image.new('RGB', (160, 100), (220, 70, 20))
    if kind == 'alpha':
        data = encoded(Image.new('RGBA', (160, 100), (255, 0, 0, 0)))
        assert rgb_from_bytes(data).getpixel((0, 0)) == (255, 255, 255)
    elif kind == 'exif':
        exif = image.getexif()
        exif[274] = 6
        data = encoded(image, 'JPEG', exif=exif)
        assert rgb_from_bytes(data).size == (100, 160)
    elif kind == 'gray':
        data = encoded(image.convert('L'))
    else:
        data = encoded(image, 'WEBP' if kind == 'webp' else 'PNG')
    actual = runtime.predict(data)
    expected = analyze_bytes(runtime.model, data, runtime.device, runtime.openset_gate)
    for key in expected:
        assert actual[key] == expected[key]
    assert actual['support_threshold'] == pytest.approx(0.700972028801349)


def test_quality_keeps_raw_predictions_and_has_separate_status(runtime):
    data = encoded(Image.new('RGB', (50, 50), 'black'))
    raw = runtime.predict(data)
    quality = runtime.predict(data, check_quality=True)
    assert quality['status'] == 'quality_rejection'
    assert not quality['supported']
    assert quality['joint_scores'] == raw['joint_scores']
    assert quality['quality']['brightness'] == 0


def test_runtime_errors_are_not_silent(runtime, monkeypatch):
    with pytest.raises(ValueError, match='TTA'):
        runtime.predict(b'', use_tta=True)
    with pytest.raises(Exception):
        runtime.predict(b'invalid image')

    def broken(*args):
        raise ValueError('gate inference failed')

    monkeypatch.setattr('freshlens_ai.inference.cnn_predict.analyze_bytes', broken)
    with pytest.raises(ValueError, match='gate inference failed'):
        runtime.predict(encoded(Image.new('RGB', (100, 100))))


def test_missing_and_mismatched_gate_fail(tmp_path):
    model = PROJECT_DIR / 'models/cnn_efficientnet_b0/best.pt'
    with pytest.raises(Exception, match='gate'):
        FreshLensPredictor(model, tmp_path / 'absent.npz', device='cpu')
    meta = tmp_path / 'gate.json'
    meta.write_text(json.dumps({'checkpoint_sha256': 'wrong'}))
    with pytest.raises(Exception, match='best.pt'):
        FreshLensPredictor(model, model.with_name('open_set_gate.npz'), 'cpu', gate_meta_path=meta)


def record(correct=True, accepted=True, known=True):
    return dict(known=known, true_joint='apple::fresh' if known else 'other',
                true_fruit='apple' if known else 'other', true_condition='fresh',
                fruit='apple', condition='fresh' if correct else 'rotten',
                joint_class='apple::fresh' if correct else 'apple::rotten',
                supported=accepted, gate_supported=accepted,
                status='accepted' if accepted else 'openset_rejection',
                joint_confidence=0.8, quality_passed=True, latency_ms=10)


def test_denominators_and_all_four_gate_outcomes():
    rows = [record(c, a) for c in (True, False) for a in (True, False)]
    rows += [record(known=False), record(known=False, accepted=False)]
    result = summarize(rows)
    assert result['total_known'] == 4
    assert result['cnn_only']['joint']['accuracy'] == 0.5
    assert result['policy']['known_coverage'] == 0.5
    assert result['policy']['end_to_end_joint_success'] == 0.25
    assert result['policy']['accepted_only_joint_accuracy'] == 0.5
    assert result['policy']['unknown_false_acceptance_rate'] == 0.5
    assert all(n == 1 for n in result['gate_only']['known_joint_outcomes'].values())
    assert len(result['cnn_only']['joint']['confusion_matrix']) == 8
    assert summarize([record(accepted=False)])['policy']['accepted_only_joint_accuracy'] is None
    assert summarize([record()])['policy']['unknown_rejection_rate'] is None
    assert summarize([record(known=False)])['cnn_only']['joint'] is None


def test_manifest_and_windows_names(tmp_path):
    from EVALUATE_TV3 import load_rows, gallery_name
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('path,fruit,status,split,group_id\na.png,apple,fresh,test,g\nb.png,apple,fresh,train,g\n')
    with pytest.raises(ValueError, match='crosses splits'):
        load_rows(manifest, 'test')
    name = gallery_name(1, {'fruit': 'apple', 'status': 'rotten'},
                        {'joint_class': 'apple::fresh', 'status': 'accepted'})
    assert not any(char in name for char in '<>:"/\\|?*')


def test_gradcam_uses_fruit_first_and_removes_hooks():
    # Raw argmax=banana fresh (0.35), but apple marginal=0.55 is largest.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Conv2d(3, 1, 1)

        def forward(self, tensor):
            feature = self.features(tensor).mean((1, 2, 3))[:, None]
            return feature + torch.tensor([.30, .25, .35, .01, .03, .02, .02, .02]).log()

    model = Toy()
    hooks = len(model.features._forward_hooks)
    with GradCAM(model, model.features) as cam:
        heatmap, selected = cam.generate_heatmap(torch.ones(1, 3, 8, 8))
        assert selected == 0
        assert np.isfinite(heatmap).all()
    assert len(model.features._forward_hooks) == hooks


def test_evaluator_outputs_complete_report(runtime, tmp_path, monkeypatch):
    import EVALUATE_TV3 as evaluation
    from GENERATE_GRADCAM_DEMO import run as cam_run
    from argparse import Namespace
    image = tmp_path / 'photo.png'
    image.write_bytes(encoded(Image.new('RGB', (160, 100), 'red')))
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('path,fruit,status,split,group_id,quality_label\nphoto.png,apple,fresh,dev,a,bad\n')
    monkeypatch.setattr(evaluation, 'FreshLensPredictor', lambda *a, **kw: runtime)
    args = evaluation.build_parser().parse_args(['--root', str(tmp_path), '--manifest', str(manifest),
        '--split', 'dev', '--output', str(tmp_path / 'results'), '--quality'])
    report = evaluation.run(args)
    assert report['metrics']['total_known'] == 1
    assert report['metrics']['policy']['known_quality_rejected'] == 1
    assert report['metrics']['cnn_only']['joint']['per_class']['apple::fresh']['support'] == 1
    for filename in ['metrics.json', 'predictions.csv', 'confusion_matrix.png', 'manifest.csv']:
        assert (args.output / filename).is_file()
    with (args.output / 'predictions.csv').open(encoding='utf-8-sig') as f:
        row = next(csv.DictReader(f))
        assert (args.output / row['gallery_file']).is_file()
    cam_run(Namespace(image=image, checkpoint=PROJECT_DIR/'models/cnn_efficientnet_b0/best.pt',
                      output=tmp_path/'cam.png', device='cpu'))
    assert Image.open(tmp_path/'cam.png').size == (224, 224)
    assert json.loads((tmp_path/'cam.json').read_text())['target_class'] == runtime.predict(image)['joint_class']


def test_quality_configuration_validation():
    with pytest.raises(ValueError):
        QualityConfig(dark_threshold=230, bright_threshold=200)


def analyze_button(app):
    """Locate the Analyze button, not Nhi's new clear-image '×' button."""
    matching = [button for button in app.button if button.label == 'Phân tích ảnh']
    assert len(matching) == 1, (
        'Expected one Analyze button; found: '
        + repr([button.label for button in app.button])
    )
    return matching[0]


def test_streamlit_analysis_and_state_reset(runtime, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    upload = [encoded(Image.new('RGB', (50, 50), 'black'))]
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: io.BytesIO(upload[0]))
    monkeypatch.setattr('freshlens_ai.inference.cnn_predict.FreshLensPredictor', lambda *a, **kw: runtime)
    app = AppTest.from_file(str(PROJECT_DIR / 'APP_CNN_V2.py'), default_timeout=30).run()
    assert not app.exception
    analyze_button(app).click().run()
    assert not app.exception
    assert app.session_state['last_result']['status'] == 'openset_rejection'
    app.checkbox[0].check().run()
    assert 'last_result' not in app.session_state
    analyze_button(app).click().run()
    assert not app.exception
    assert app.session_state['last_result']['status'] == 'quality_rejection'
    # The warning title is styled Markdown; st.warning contains the reason only.
    assert app.warning
    assert any('Ảnh chưa đạt chất lượng' in block.value for block in app.markdown)
    upload[0] = encoded(Image.new('RGB', (50, 50), 'white'))
    app.run()
    assert 'last_result' not in app.session_state
    assert not app.warning  # The previous image's quality warning is hidden.
    upload[0] = b'broken image'
    app.run()
    assert not app.exception
    assert app.error
    assert 'last_result' not in app.session_state


@pytest.mark.parametrize('field,value', [
    ('decision_threshold', np.array([-1.])),
    ('decision_threshold', np.array([1.1])),
    ('decision_threshold', np.array([np.nan])),
    ('decision_threshold', np.array([0.5])),  # Valid number, mismatched JSON.
    ('decision_threshold', np.array([0.7, 0.8])),
    ('scaler_scale', np.zeros(6)),
    ('scaler_scale', -np.ones(6)),
    ('coef', np.ones(5)),
    ('coef', np.full(6, np.inf)),
    ('prototypes', np.ones((4, 4, 10))),
    ('prototypes', np.zeros((4, 4, 1280))),
])
def test_reject_malformed_gate(tmp_path, field, value):
    from freshlens_ai.inference.open_set import load_gate
    from freshlens_ai.errors import DataError
    model = PROJECT_DIR / 'models/cnn_efficientnet_b0/best.pt'
    with np.load(model.with_name('open_set_gate.npz')) as archive:
        data = {key: archive[key] for key in archive.files}
    data[field] = value
    path = tmp_path / 'gate.npz'
    np.savez(path, **data)
    with pytest.raises(DataError):
        load_gate(path, model.with_name('open_set_gate.json'), model)


@pytest.mark.parametrize('values', [
    {},
    dict(min_size=1.5, blur_threshold=0, dark_threshold=0, bright_threshold=255),
    dict(min_size=100, blur_threshold='100', dark_threshold=40, bright_threshold=215),
    dict(min_size=100, blur_threshold=float('nan'), dark_threshold=40, bright_threshold=215),
])
def test_invalid_quality_file(tmp_path, values):
    from freshlens_ai.inference.quality import load_quality_config
    path = tmp_path / 'quality.json'
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_quality_config(path)


def test_quality_config_shared_by_cli_evaluator_and_app(runtime, tmp_path, monkeypatch):
    import streamlit as st
    import PREDICT_CNN_V2 as cli
    import EVALUATE_TV3 as evaluator
    from dataclasses import asdict
    from streamlit.testing.v1 import AppTest
    from freshlens_ai.inference.quality import load_quality_config, quality_config_hash

    config_path = tmp_path / 'quality.json'
    custom = QualityConfig(min_size=10, blur_threshold=0, dark_threshold=0, bright_threshold=255)
    config_path.write_text(json.dumps(asdict(custom)))
    photo = tmp_path / 'photo.png'
    photo.write_bytes(encoded(Image.new('RGB', (50, 50), 'black')))
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('path,fruit,status,split\nphoto.png,apple,fresh,dev\n')
    observed = []

    def factory(*args, **kwargs):
        observed.append(kwargs['quality_config'])
        monkeypatch.setattr(runtime, 'quality_config', kwargs['quality_config'])
        return runtime

    monkeypatch.setattr(cli, 'FreshLensPredictor', factory)
    monkeypatch.setattr(evaluator, 'FreshLensPredictor', factory)
    result = cli.run(cli.build_parser().parse_args([
        '--image', str(photo), '--quality', '--quality-config', str(config_path)]))
    assert result['quality']['passed']
    args = evaluator.build_parser().parse_args([
        '--root', str(tmp_path), '--manifest', str(manifest), '--split', 'dev',
        '--quality', '--quality-config', str(config_path), '--output', str(tmp_path/'output')])
    report = evaluator.run(args)
    saved = args.output / 'quality_config.json'
    assert load_quality_config(saved) == custom
    assert report['quality_config_sha256'] == result['quality_config_sha256'] == quality_config_hash(custom)
    monkeypatch.setenv('FRESHLENS_QUALITY_CONFIG', str(saved))
    monkeypatch.setattr('freshlens_ai.inference.cnn_predict.FreshLensPredictor', factory)
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **kw: io.BytesIO(photo.read_bytes()))
    st.cache_resource.clear()
    app = AppTest.from_file(str(PROJECT_DIR/'APP_CNN_V2.py'), default_timeout=30).run()
    assert not app.exception
    app.checkbox[0].check().run()
    assert not app.exception
    analyze_button(app).click().run()
    assert not app.exception
    assert app.session_state['last_result']['quality_config_sha256'] == result['quality_config_sha256']
    # Editing the config resets the existing result on the next Streamlit rerun.
    saved.write_text(json.dumps(asdict(QualityConfig())))
    app.run()
    assert not app.exception
    assert 'last_result' not in app.session_state
    assert observed[:3] == [custom, custom, custom]
    st.cache_resource.clear()
