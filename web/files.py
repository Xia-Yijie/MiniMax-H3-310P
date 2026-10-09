"""Workspace file operations; never follow links outside the chosen root."""
import json
from datetime import datetime
import shutil
from pathlib import Path
from flask import jsonify, request, send_file
from .backend import workspace, JOB_ID
from infer import server as resident


def register_files(app, engine):
    def root():
        data = request.get_json(silent=True) or {}
        return workspace(request.values.get('work_dir') or data.get('work_dir') or app.config['DEFAULT_WORK_DIR'])

    @app.get('/api/folders')
    def folders():
        path = Path(request.args.get('path') or Path.home()).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError('文件夹不存在')
        entries = []
        for item in path.iterdir():
            try:
                if item.is_dir() and not item.is_symlink():
                    entries.append(dict(name=item.name, path=str(item)))
            except OSError:
                continue
        entries.sort(key=lambda entry: entry['name'].lower())
        return jsonify(path=str(path), parent=str(path.parent), home=str(Path.home()), entries=entries)

    def target(base, value, allow_root=False):
        relative = Path(value or '.')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('文件路径必须位于工作目录内')
        path = base / relative
        current = base
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('不支持操作符号链接')
        if not path.resolve().is_relative_to(base) or (path == base and not allow_root):
            raise ValueError('不能操作工作目录本身')
        return path

    def projects(base):
        records = []
        for folder in (base / 'jobs').iterdir():
            if folder.is_symlink() or not folder.is_dir() or not JOB_ID.fullmatch(folder.name):
                continue
            record_path = folder / 'request.json'
            if not record_path.is_file():
                continue
            target(base, str(record_path.relative_to(base)))
            try:
                record = json.loads(record_path.read_text())
                linked = set()
                for values in record.get('inputs', {}).values():
                    for value in values:
                        linked.add(target(base, str(Path(value).relative_to(base))))
                if record.get('output'):
                    output = target(base, str(Path(record['output']).relative_to(base)))
                    linked.add(output)
                    linked.add(output.with_suffix('.pixel.json'))
                records.append((folder, record, linked))
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError('项目记录无效，无法安全管理文件：' + folder.name) from error
        return records

    def overlap(left, right):
        return left == right or left.is_relative_to(right) or right.is_relative_to(left)

    def association(path, records):
        return any(overlap(path, folder) or any(overlap(path, linked) for linked in assets)
                   for folder, _, assets in records)

    def basic_mutable(path, base):
        if path == base or path in [base / n for n in ('inputs', 'outputs', 'jobs')]:
            raise ValueError('不能重命名或删除工作目录的基础文件夹')
        if engine.status().get('state') == 'busy':
            raise ValueError('正在生成视频，完成后再重命名或删除文件')

    def mutable(path, base):
        basic_mutable(path, base)
        if association(path, projects(base)):
            raise ValueError('文件与项目关联，不能单独重命名或删除；请从任务目录删除项目')

    @app.get('/api/files')
    def listing():
        base = root(); path = target(base, request.args.get('path'), True)
        if not path.is_dir():
            raise FileNotFoundError('文件夹不存在')
        entries = []
        records = projects(base)
        for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if path == base and item.name not in ('inputs', 'jobs', 'outputs'):
                continue
            if item.is_symlink():
                continue
            project = next((r for f, r, _ in records if f == item), None)
            protected = item in [base / n for n in ('inputs', 'outputs', 'jobs')]
            linked = association(item, records)
            display_name = {'inputs': '输入', 'jobs': '任务', 'outputs': '输出'}.get(item.name) if path == base else None
            related = next((r for f, r, assets in records if f == item or
                            (path == base / 'inputs' and any(asset.is_relative_to(item) for asset in assets)) or
                            (path == base / 'outputs' and item in assets)), None)
            if related is not None:
                prompt = ' '.join(str(related.get('parameters', {}).get('prompt', '未命名视频')).split())
                title = prompt[:22] + ('…' if len(prompt) > 22 else '')
                stamp = datetime.fromtimestamp(related.get('created', 0)).strftime('%m-%d %H:%M')
                display_name = f'{title} · {stamp}'
                if path == base / 'outputs' and item.suffix == '.json':
                    display_name += ' · 超分报告'
            elif path.parent == base / 'inputs':
                display_name = {'first_frame_1':'首帧', 'last_frame_1':'尾帧'}.get(item.stem)
                if item.stem.startswith('references_') and item.stem.split('_')[-1].isdigit():
                    display_name = '参考图 ' + item.stem.split('_')[-1]
                if display_name:
                    display_name += item.suffix
            if path.parent == base / 'jobs':
                display_name = {'request.json':'任务参数.json', 'source.mp4':'原始视频.mp4',
                                'source.progress.jsonl':'生成进度.jsonl', 'source.json':'生成报告.json',
                                'source.latents.json':'潜变量信息.json', 'source.latents.npz':'潜变量缓存.npz'}.get(item.name, display_name)
            entries.append(dict(name=item.name, display_name=display_name, path=str(item.relative_to(base)),
                                directory=item.is_dir(), size=item.stat().st_size if item.is_file() else None,
                                project=project is not None, project_state=project.get('state') if project else None, linked=linked, deletable=not protected and (project is not None or not linked),
                                renamable=not protected and not linked))
        return jsonify(entries=entries, path=str(path.relative_to(base)))

    @app.get('/api/files/content')
    def content():
        base = root(); path = target(base, request.args.get('path'))
        if not path.is_file():
            raise FileNotFoundError('文件不存在')
        media = path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm', '.mov')
        response = send_file(path, conditional=True, as_attachment=not media or request.args.get('download') == '1')
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.post('/api/files/upload')
    def upload():
        base = root(); folder = target(base, request.form.get('path'), True)
        if not folder.is_dir():
            raise FileNotFoundError('文件夹不存在')
        uploads = request.files.getlist('files')
        if not uploads:
            raise ValueError('请选择文件')
        pending = []
        for item in uploads:
            name = Path(item.filename or '').name
            if not name or name in ('.', '..') or '/' in name or '\\' in name:
                raise ValueError('文件名无效')
            dest = target(base, str((folder / name).relative_to(base)))
            if dest.exists() or any(dest == p for p, _ in pending):
                raise ValueError('同名文件已存在：' + name)
            pending.append((dest, item))
        for dest, item in pending:
            with dest.open('xb') as stream:
                shutil.copyfileobj(item.stream, stream)
        return jsonify(uploaded=[str(p.relative_to(base)) for p, _ in pending])

    @app.post('/api/files/rename')
    def rename():
        data = request.get_json() or {}; base = root(); path = target(base, data.get('path'))
        mutable(path, base)
        name = data.get('name', '').strip()
        if not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('请输入有效的文件名')
        dest = target(base, str((path.parent / name).relative_to(base)))
        if dest.exists():
            raise ValueError('同名文件已存在')
        path.rename(dest)
        return jsonify(path=str(dest.relative_to(base)))

    @app.post('/api/files/delete')
    def delete():
        data = request.get_json() or {}; base = root(); path = target(base, data.get('path'))
        basic_mutable(path, base)
        records = projects(base)
        project = next(((r, assets) for folder, r, assets in records if folder == path), None)
        if project:
            if data.get('confirm_project') is not True:
                raise ValueError('删除项目需要二次确认')
            record, assets = project
            queue = resident.state_dir(record.get('device', 1))
            if any((queue / state / (path.name + '.json')).exists() for state in ('pending', 'running')):
                raise ValueError('项目正在排队或生成，暂时不能删除')
            removals = []
            if data.get('delete_assets') is True:
                others = [entry for entry in records if entry[0] != path]
                # Shared assets remain protected by their other projects.
                removals = [asset for asset in assets if not association(asset, others) and asset.is_file()]
            for asset in removals:
                asset.unlink()
            shutil.rmtree(path)
            # Remove only empty input folders, never unrelated uploaded files.
            for asset in removals:
                parent = asset.parent
                while parent.is_relative_to(base / 'inputs') and parent != base / 'inputs':
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            return jsonify(deleted=True, project=True, deleted_assets=len(removals))
        if association(path, records):
            raise ValueError('文件与项目关联，不能单独删除；请从任务目录删除项目')
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return jsonify(deleted=True)
