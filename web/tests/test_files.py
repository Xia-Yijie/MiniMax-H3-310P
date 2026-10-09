import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from web.app import create_app


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'work'
        self.engine = Mock()
        self.engine.status.return_value = {'state': 'ready'}
        self.client = create_app(self.root, self.engine).test_client()
        self.client.get('/api/files')

    def test_upload_list_download_rename_delete(self):
        r = self.client.post('/api/files/upload', data={'path': 'inputs', 'files': (io.BytesIO(b'hello'), '你好.txt')})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get('/api/files?path=inputs').get_json()['entries'][0]['name'], '你好.txt')
        r = self.client.get('/api/files/content?path=inputs/你好.txt&download=1')
        self.assertEqual(r.data, b'hello'); r.close()
        self.assertEqual(self.client.post('/api/files/rename', json={'path': 'inputs/你好.txt', 'name': 'new.txt'}).status_code, 200)
        self.assertEqual(self.client.post('/api/files/delete', json={'path': 'inputs/new.txt'}).status_code, 200)
        self.assertFalse((self.root / 'inputs/new.txt').exists())

    def test_traversal_symlink_and_root_are_rejected(self):
        outside = Path(self.temp.name) / 'outside'; outside.write_text('private')
        (self.root / 'link').symlink_to(outside)
        for path in ('../outside', str(outside), 'link'):
            self.assertEqual(self.client.get('/api/files/content', query_string={'path': path}).status_code, 400)
            self.assertEqual(self.client.post('/api/files/delete', json={'path': path}).status_code, 400)
        self.assertEqual(self.client.post('/api/files/delete', json={'path': '.'}).status_code, 400)
        self.assertEqual(self.client.post('/api/files/delete', json={'path': 'inputs'}).status_code, 400)
        self.assertTrue(outside.exists())

    def test_upload_does_not_overwrite_and_html_is_download_only(self):
        (self.root/'page.html').write_text('<script>alert(1)</script>')
        r = self.client.post('/api/files/upload', data={'files': (io.BytesIO(b'overwrite'), 'page.html')})
        self.assertEqual(r.status_code, 400)
        r = self.client.get('/api/files/content?path=page.html')
        self.assertIn('attachment', r.headers['Content-Disposition']); r.close()

    def test_busy_guard_and_nonempty_directory_delete(self):
        (self.root/'sample').mkdir()
        (self.root/'sample/file.txt').write_text('data')
        self.engine.status.return_value = {'state': 'busy'}
        self.assertEqual(self.client.post('/api/files/delete', json={'path': 'sample'}).status_code, 400)
        self.assertEqual(self.client.post('/api/files/rename', json={'path': 'sample', 'name': 'changed'}).status_code, 400)
        self.engine.status.return_value = {'state': 'ready'}
        self.assertEqual(self.client.post('/api/files/delete', json={'path': 'sample'}).status_code, 200)
        self.assertFalse((self.root/'sample').exists())

    def make_project(self, index=1, shared_input=None):
        import json
        identifier = f'{index:020d}_' + f'{index:032x}'
        folder = self.root / 'jobs' / identifier; folder.mkdir()
        source = shared_input or self.root / 'inputs' / identifier / 'ref.png'
        source.parent.mkdir(exist_ok=True); source.write_bytes(b'image')
        output = self.root / 'outputs' / (identifier + '.mp4'); output.write_bytes(b'video')
        output.with_suffix('.pixel.json').write_text('{}')
        (folder/'request.json').write_text(json.dumps(dict(id=identifier, device=1, state='done',
            inputs={'--reference-image':[str(source)]}, output=str(output))))
        return folder, source, output

    def test_top_level_has_only_three_named_folders(self):
        (self.root/'extra').mkdir();(self.root/'other.txt').write_text('x')
        entries=self.client.get('/api/files').get_json()['entries']
        self.assertEqual({e['display_name'] for e in entries},{'输入','任务','输出'})
        self.assertTrue(all(not e['deletable'] and not e['renamable'] for e in entries))

    def test_linked_files_and_ancestor_folders_are_protected(self):
        folder, source, output = self.make_project()
        for path in (source,source.parent,output,output.with_suffix('.pixel.json'),folder/'request.json'):
            relative=str(path.relative_to(self.root))
            self.assertEqual(self.client.post('/api/files/delete',json={'path':relative}).status_code,400)
            self.assertEqual(self.client.post('/api/files/rename',json={'path':relative,'name':'renamed'}).status_code,400)
        self.assertTrue(source.exists());self.assertTrue(output.exists())
        self.assertEqual(self.client.post('/api/files/delete',json={'path':str(folder.relative_to(self.root))}).status_code,400)

    def test_delete_project_retains_assets_by_default(self):
        folder, source, output = self.make_project()
        r=self.client.post('/api/files/delete',json={'path':str(folder.relative_to(self.root)),'confirm_project':True})
        self.assertEqual(r.status_code,200,r.get_json())
        self.assertFalse(folder.exists());self.assertTrue(source.exists());self.assertTrue(output.exists())
        self.assertEqual(self.client.post('/api/files/delete',json={'path':str(source.relative_to(self.root))}).status_code,200)

    def test_delete_project_cleans_own_assets_but_preserves_shared(self):
        folder, source, output = self.make_project()
        second, _, other_output = self.make_project(2,source)
        r=self.client.post('/api/files/delete',json={'path':str(folder.relative_to(self.root)),
            'confirm_project':True,'delete_assets':True})
        self.assertEqual(r.status_code,200,r.get_json());self.assertFalse(folder.exists())
        self.assertTrue(source.exists());self.assertTrue(other_output.exists());self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.pixel.json').exists())
        r=self.client.post('/api/files/delete',json={'path':str(second.relative_to(self.root)),
            'confirm_project':True,'delete_assets':True})
        self.assertEqual(r.status_code,200);self.assertFalse(source.exists());self.assertFalse(source.parent.exists())

    def test_queued_project_cannot_be_deleted(self):
        from unittest.mock import patch
        folder, source, output = self.make_project()
        queue = Path(self.temp.name)/'queue';(queue/'pending').mkdir(parents=True)
        (queue/'pending'/(folder.name+'.json')).write_text('{}')
        with patch('infer.server.state_dir',return_value=queue):
            r=self.client.post('/api/files/delete',json={'path':str(folder.relative_to(self.root)),
                'confirm_project':True,'delete_assets':True})
        self.assertEqual(r.status_code,400);self.assertTrue(folder.exists());self.assertTrue(source.exists())

    def test_projects_and_assets_have_readable_display_names(self):
        import json
        folder, source, output = self.make_project()
        record_path=folder/'request.json'
        record=json.loads(record_path.read_text())
        record['parameters']={'prompt':'参考图中的女子跳舞'}
        record['created']=1791562387
        record_path.write_text(json.dumps(record))
        for directory in ('jobs','inputs','outputs'):
            entries=self.client.get('/api/files',query_string={'path':directory}).get_json()['entries']
            self.assertIn('参考图中的女子跳舞',entries[0]['display_name'])
            self.assertNotIn(folder.name,entries[0]['display_name'])
        entries=self.client.get('/api/files',query_string={'path':str(folder.relative_to(self.root))}).get_json()['entries']
        self.assertEqual(entries[0]['display_name'],'任务参数.json')
