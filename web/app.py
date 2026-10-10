"""Run with python -m web.app; the NPU model lives in the existing worker."""
import argparse
import io
import threading
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired

from flask import Flask, g, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from .backend import Backend, DEFAULT_WORK_DIR, new_identifier, read_job, validate_parameters, workspace


def create_app(default_work_dir=None, backend=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024**2
    app.config['DEFAULT_WORK_DIR'] = str(default_work_dir or DEFAULT_WORK_DIR)
    engine = backend or Backend()
    file_lock = threading.RLock()

    @app.before_request
    def serialize_workspace_changes():
        if request.path.startswith(('/api/files', '/api/jobs')):
            file_lock.acquire()
            g.workspace_locked = True

    @app.teardown_request
    def release_workspace_lock(error):
        if getattr(g, 'workspace_locked', False):
            g.workspace_locked = False
            file_lock.release()

    @app.before_request
    def same_origin():
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
                return jsonify(error='请从本页面提交请求'), 403

    @app.errorhandler(Exception)
    def error_response(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        if isinstance(error, FileNotFoundError):
            return jsonify(error=str(error)), 404
        if isinstance(error, (ValueError, OSError, RuntimeError, TimeoutExpired)):
            return jsonify(error=str(error)), 400
        app.logger.exception('Web request failed')
        return jsonify(error='页面服务内部错误，请查看终端日志'), 500

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/config')
    def config():
        return jsonify(work_dir=app.config['DEFAULT_WORK_DIR'])

    @app.post('/api/workspace')
    def select_workspace():
        value = (request.get_json(silent=True) or {}).get('work_dir')
        if not value:
            raise ValueError('请选择工作目录')
        return jsonify(work_dir=str(workspace(value)))

    @app.get('/api/status')
    def status():
        return jsonify(engine.status())

    @app.post('/api/start')
    def start():
        return jsonify(engine.start())

    def selected_root():
        return workspace(request.values.get('work_dir') or app.config['DEFAULT_WORK_DIR'])

    @app.post('/api/jobs')
    def submit():
        parameters = validate_parameters(request.form)
        uploads = {key: [f for f in request.files.getlist(key) if f.filename]
                   for key in ('first_frame', 'last_frame', 'references', 'reference_videos')}
        if len(uploads['first_frame']) > 1 or len(uploads['last_frame']) > 1:
            raise ValueError('首帧和尾帧各只能上传一张图')
        if len(uploads['references']) > 8:
            raise ValueError('最多上传 8 张参考图')
        if len(uploads['reference_videos']) > 3:
            raise ValueError('最多上传 3 段参考视频')
        mode = parameters['mode']
        if mode == 'text' and any(uploads.values()):
            raise ValueError('文字模式不接受参考素材')
        if mode == 'frames' and (uploads['references'] or uploads['reference_videos'] or not (uploads['first_frame'] or uploads['last_frame'])):
            raise ValueError('首尾帧模式至少需要一张首帧或尾帧图，不能混入参考图片或视频')
        if mode == 'references' and (uploads['first_frame'] or uploads['last_frame'] or not (uploads['references'] or uploads['reference_videos'])):
            raise ValueError('参考模式至少需要一张图片或一段视频，不能混入首尾帧')
        # Validate every image before creating files or starting the model.
        prepared = []
        video_seconds = 0
        extensions = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}
        for key, files in uploads.items():
            for index, upload in enumerate(files):
                if key == 'reference_videos':
                    from infer.media import video_spec
                    suffix = Path(upload.filename).suffix.lower()
                    if suffix not in ('.mp4','.webm','.mov','.mkv'):
                        raise ValueError('参考视频支持 MP4、WebM、MOV、MKV')
                    data = upload.stream.read(96 * 1024**2 + 1)
                    if len(data)>96*1024**2:raise ValueError('每段参考视频不能超过 96 MiB')
                    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                        temporary.write(data);temporary.flush()
                        spec=video_spec(temporary.name,parameters['frames'])
                    video_seconds += spec['frames']/24
                    if video_seconds>15:raise ValueError('参考视频总时长不能超过 15 秒')
                    prepared.append((key,index,suffix,data))
                    continue
                data = upload.stream.read(20 * 1024**2 + 1)
                if len(data) > 20 * 1024**2:
                    raise ValueError('每张参考图不能超过 20 MiB')
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        suffix = extensions.get(image.format)
                        if not suffix or image.width * image.height > 25_000_000:
                            raise ValueError('请上传不超过 2500 万像素的 PNG、JPEG 或 WebP 图片')
                        image.verify()
                except (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError) as error:
                    raise ValueError('无法读取参考图，请重新选择图片') from error
                prepared.append((key, index, suffix, data))
        root = selected_root()
        identifier = new_identifier()
        images = {'--first-frame': [], '--last-frame': [], '--reference-image': [], '--reference-video': []}
        flags = dict(first_frame='--first-frame', last_frame='--last-frame', references='--reference-image',reference_videos='--reference-video')
        for key, index, suffix, data in prepared:
            folder = root / 'inputs' / identifier
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{key}_{index + 1}{suffix}'
            path.write_bytes(data)
            images[flags[key]].append(path)
        record = engine.submit(root, parameters, images, identifier)
        return jsonify(record), 202

    @app.get('/api/jobs')
    def history():
        root = selected_root()
        paths = sorted((root / 'jobs').glob('*/request.json'), reverse=True)[:100]
        return jsonify(jobs=[read_job(root, path.parent.name) for path in paths])

    @app.get('/api/jobs/<identifier>')
    def job(identifier):
        return jsonify(read_job(selected_root(), identifier))

    @app.get('/api/jobs/<identifier>/video')
    def video(identifier):
        root = selected_root()
        record = read_job(root, identifier)
        if not record['video_available']:
            raise FileNotFoundError('视频尚未生成或已被移走')
        path = Path(record['output']).resolve()
        if not path.is_relative_to((root / 'outputs').resolve()):
            raise ValueError('输出路径不属于当前工作目录')
        return send_file(path, mimetype='video/mp4', conditional=True,
                         as_attachment=request.args.get('download') == '1')

    from .files import register_files
    register_files(app, engine)
    return app


def main():
    parser = argparse.ArgumentParser(description='MiniMax-H3 本地推理页面')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19648)
    parser.add_argument('--device', type=int, choices=(0, 1), default=1)
    parser.add_argument('--work-dir', default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()
    app = create_app(args.work_dir, Backend(args.device))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
