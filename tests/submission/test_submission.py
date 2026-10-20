"""Submission checks use synthetic images, never accuracy claims."""
import csv
import io
import json
import numpy as np
import pytest
from PIL import Image
from PREPARE_CAMERA_MANIFEST import prepare
from BENCHMARK_MODELS import load_candidates
from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference.app_state import clear_result


def labels(tmp_path, **changes):
    Image.new('RGB', (130, 100), 'red').save(tmp_path/'photo.png')
    row = dict(path='photo.png', fruit='apple', status='fresh', split='dev',
               group_id='camera_001', source='phone', quality_label='good')
    row.update(changes)
    path = tmp_path/'labels.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_camera_provenance_and_quality(tmp_path):
    path = labels(tmp_path)
    report = prepare(tmp_path, path, tmp_path/'out')
    assert report['images'] == 1 and not report['independent_final_test_verified']
    assert json.loads((tmp_path/'out/quality.json').read_text(encoding='utf-8'))[0]['width'] == 130
    assert len(report['manifest_sha256']) == 64
    with pytest.raises(ValueError, match='already exists'):
        prepare(tmp_path, path, tmp_path/'out')


@pytest.mark.parametrize('changes', [dict(group_id=''), dict(fruit='mango'),
                                    dict(status='bad'), dict(split='train'),
                                    dict(path='../outside.png')])
def test_camera_rejects_invalid_inputs(tmp_path, changes):
    path = labels(tmp_path, **changes)
    with pytest.raises(Exception):
        prepare(tmp_path, path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_camera_rejects_training_overlap(tmp_path):
    path = labels(tmp_path)
    reference = tmp_path/'training.csv'
    reference.write_text('sha256,group_id\n,camera_001\n')
    with pytest.raises(ValueError, match='overlaps reference'):
        prepare(tmp_path, path, tmp_path/'out', reference)


def test_benchmark_validation(tmp_path):
    model = PROJECT_DIR/'models/cnn_efficientnet_b0'
    item = dict(name='baseline', checkpoint=str(model/'best.pt'),
                gate_npz=str(model/'open_set_gate.npz'), gate_meta=str(model/'open_set_gate.json'))
    path = tmp_path/'candidates.json'
    path.write_text(json.dumps([item]))
    assert load_candidates(path)[0]['name'] == 'baseline'
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match='unique'):
        load_candidates(path)
    item['name'] = '../unsafe'
    path.write_text(json.dumps([item]))
    with pytest.raises(ValueError, match='names'):
        load_candidates(path)


def test_live_gradcam_preserves_prediction():
    import torch
    from freshlens_ai.inference.cnn_predict import FreshLensPredictor
    torch.set_num_threads(2)
    runtime = FreshLensPredictor(PROJECT_DIR/'models/cnn_efficientnet_b0/best.pt', device='cpu')
    buffer = io.BytesIO()
    Image.new('RGB', (180, 100), 'orange').save(buffer, format='PNG')
    data = buffer.getvalue()
    before = runtime.predict(data)
    count = len(runtime.model.features[-1]._forward_hooks)
    explanation = runtime.explain(data)
    assert before['joint_scores'] == runtime.predict(data)['joint_scores']
    assert explanation['target_class'] == before['joint_class']
    assert explanation['overlay'].shape == (224, 224, 3)
    assert np.isfinite(explanation['heatmap']).all()
    assert len(runtime.model.features[-1]._forward_hooks) == count
    state = {'image_explanation': explanation, 'last_result': before}
    clear_result(state)
    assert state == {}


def test_invalid_upload_clears_result(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    upload = io.BytesIO(b'broken-image')
    upload.name = 'broken.jpg'
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **kw: upload)
    app = AppTest.from_string(
        'from ui.diagnosis import render_diagnosis_page\n'
        'import streamlit as st\n'
        'st.session_state["last_result"] = {"fruit": "apple"}\n'
        'render_diagnosis_page(object())'
    ).run()
    assert not app.exception
    assert any('Không đọc được ảnh' in error.value for error in app.error)
    assert app.session_state['current_image_bytes'] is None
    assert 'last_result' not in app.session_state


@pytest.mark.parametrize('page', ['About', 'Giải thích', 'Đánh giá'])
def test_pages(page):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(PROJECT_DIR/'APP_CNN_V2.py'), default_timeout=30)
    app.query_params['page'] = page
    app.run()
    assert not app.exception
    if page == 'About':
        text = '\n'.join(m.value for m in app.markdown)
        for code in ['N23DCCN096', 'N23DCCN078', 'N23DCCN118', 'N23DCCN112']:
            assert code in text
