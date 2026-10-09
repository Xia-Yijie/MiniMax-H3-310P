import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from web.app import create_app
from web.backend import Backend, DEFAULT_WORK_DIR, read_job


def image_bytes():
    data = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(data, format='PNG')
    return data.getvalue()


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        self.queue = Path(self.temp.name) / 'queue'
        self.queue.mkdir()
        for name in ('pending', 'running', 'done', 'failed'):
            (self.queue / name).mkdir()
        for target, value in [('infer.server.state_dir', self.queue), ('infer.server.worker_alive', True)]:
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        self.app = create_app(self.root, Backend())
        self.client = self.app.test_client()

    def submit(self, **changes):
        values = dict(prompt='A red ball rolls.', mode='text', seconds='1',
                      width='1280', height='720', seed='42', work_dir=str(self.root))
        values.update(changes)
        return self.client.post('/api/jobs', data=values)

    def test_default_and_page(self):
        default = create_app().test_client().get('/api/config').get_json()
        self.assertEqual(default['work_dir'], str(DEFAULT_WORK_DIR))
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('工作目录'.encode(), response.data)

    def test_text_job_is_persisted_and_queued_in_selected_directory(self):
        response = self.submit()
        self.assertEqual(response.status_code, 202, response.get_json())
        record = response.get_json()
        queued = json.loads((self.queue / 'pending' / (record['id'] + '.json')).read_text())
        argv = queued['args']
        self.assertIn('--backbone-family', argv)
        self.assertEqual(argv[argv.index('--backbone-family') + 1], 'hybrid')
        self.assertEqual(Path(queued['pixel_upscale']['output']).parent, self.root / 'outputs')
        self.assertEqual(Path(argv[argv.index('--output') + 1]).parent, self.root / 'jobs' / record['id'])
        self.assertTrue((self.root / 'jobs' / record['id'] / 'request.json').is_file())
        history = self.client.get('/api/jobs', query_string={'work_dir': str(self.root)}).get_json()['jobs']
        self.assertEqual(history[0]['parameters']['prompt'], 'A red ball rolls.')
        self.assertEqual(record['parameters']['frames'], 39)

    def test_original_uploaded_bytes_and_safe_names(self):
        data = image_bytes()
        response = self.submit(mode='frames', first_frame=(io.BytesIO(data), '../../escape.png'))
        self.assertEqual(response.status_code, 202, response.get_json())
        record = response.get_json()
        path = Path(record['inputs']['--first-frame'][0])
        self.assertTrue(path.is_relative_to(self.root / 'inputs'))
        self.assertEqual(path.name, 'first_frame_1.png')
        self.assertEqual(path.read_bytes(), data)
        self.assertFalse((Path(self.temp.name) / 'escape.png').exists())

    def test_reference_images_preserve_order(self):
        response = self.submit(mode='references', references=[
            (io.BytesIO(image_bytes()), 'a.png'), (io.BytesIO(image_bytes()), 'b.png')])
        self.assertEqual(response.status_code, 202, response.get_json())
        record = response.get_json()
        refs = record['inputs']['--reference-image']
        self.assertEqual([Path(p).name for p in refs], ['references_1.png', 'references_2.png'])

    def test_invalid_requests_never_queue_or_write_images(self):
        cases = [dict(prompt=' '), dict(seconds='nan'), dict(seconds='0'), dict(width='1279'),
                 dict(mode='frames'), dict(mode='references'),
                 dict(mode='text', first_frame=(io.BytesIO(image_bytes()), 'a.png')),
                 dict(mode='frames', first_frame=(io.BytesIO(b'not an image'), 'a.png')),
                 dict(mode='references', first_frame=(io.BytesIO(image_bytes()), 'a.png'),
                      references=(io.BytesIO(image_bytes()), 'b.png'))]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.submit(**case).status_code, 400)
        self.assertEqual(list((self.queue / 'pending').glob('*')), [])
        self.assertFalse((self.root / 'inputs').exists())

    def test_completion_survives_page_restart_and_video_supports_ranges(self):
        record = self.submit().get_json()
        path = Path(record['output'])
        path.write_bytes(b'0123456789')
        report = dict(output=str(path), worker_total_seconds=12.3)
        (self.queue / 'done' / (record['id'] + '.json')).write_text(json.dumps(dict(completed=True, report=report)))
        done = read_job(self.root, record['id'])
        self.assertEqual(done['state'], 'done')
        self.assertTrue(done['video_available'])
        (self.queue / 'done' / (record['id'] + '.json')).unlink()
        restarted = create_app(self.root).test_client()
        response = restarted.get(f'/api/jobs/{record["id"]}/video',
                                 query_string={'work_dir': str(self.root)}, headers={'Range': 'bytes=0-3'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b'0123')
        response.close()
        path.unlink()
        self.assertFalse(read_job(self.root, record['id'])['video_available'])

    def test_failed_job_has_no_video_even_if_partial_file_exists(self):
        record = self.submit().get_json()
        Path(record['output']).write_bytes(b'partial')
        (self.queue / 'failed' / (record['id'] + '.json')).write_text(json.dumps(dict(error='Out of memory')))
        result = read_job(self.root, record['id'])
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(result['video_available'])
        self.assertEqual(result['error'], 'Out of memory')

    def test_cross_origin_post_is_rejected(self):
        response = self.client.post('/api/workspace', json={'work_dir': str(self.root)},
                                    headers={'Origin': 'https://unrelated.example'})
        self.assertEqual(response.status_code, 403)

    def test_start_failure_is_retained_in_history_without_queueing(self):
        with patch.object(Backend, 'status', return_value={'running': False, 'state': 'stopped'}), \
                patch.object(Backend, 'start', side_effect=RuntimeError('Missing weights')):
            response = self.submit()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list((self.queue / 'pending').glob('*')), [])
        result = self.client.get('/api/jobs').get_json()['jobs'][0]
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['error'], 'Missing weights')

    def test_dead_worker_preserves_startup_failure_status(self):
        (self.queue / 'status.json').write_text(json.dumps(dict(state='startup_failed', error='CANN error')))
        with patch('infer.server.worker_alive', return_value=False):
            result = Backend().status()
        self.assertFalse(result['running'])
        self.assertEqual(result['state'], 'startup_failed')
        self.assertEqual(result['error'], 'CANN error')

    def test_other_directory_has_independent_history(self):
        self.submit()
        other = Path(self.temp.name) / 'other'
        response = self.client.post('/api/workspace', json={'work_dir': str(other)})
        self.assertEqual(response.status_code, 200)
        result = self.client.get('/api/jobs', query_string={'work_dir': str(other)}).get_json()
        self.assertEqual(result['jobs'], [])


if __name__ == '__main__':
    unittest.main()
