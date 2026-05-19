"""AutoCull - EricasZ"""
__version__ = 'dev'

import os
from PIL import Image
import importlib.util
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time, sleep
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import mimetypes
from io import BytesIO
import tempfile
import shutil
import base64
import threading
import sys
import builtins
os.environ['PYTHONUNBUFFERED'] = '1'  # 禁用Python输出缓冲，确保实时输出
log_messages = []  # 用于存储日志消息的列表
original_print = print  # 保存原始的print函数


def print_and_log(*args, **kwargs):
    """自定义print函数，同时输出到控制台和网页UI"""
    original_print(*args, **kwargs, flush=True)  # 调用原始print函数，并强制刷新缓冲区
    message = ' '.join(str(arg) for arg in args)
    log_messages.append(message)   # 将消息添加到日志列表
    if len(log_messages) > 1000:
        log_messages.pop(0)  # 限制日志数量，避免占用过多内存


builtins.print = print_and_log  # 重定向print函数

presets = {  # 预设名称 -> [{选片函数名 -> 权重}, 末位删除比例]
    # 单算法预设 - 极高速度
    '仅曝光': [{'builtin_lum': 1.0}, 0.0],
    '仅噪点': [{'builtin_noise': 1.0}, 0.0],
    '仅对焦': [{'builtin_focus': 1.0}, 0.0],
    '仅色彩': [{'builtin_color': 1.0}, 0.0],
    '仅 useraes': [{'ericasz_useraes': 1.0}, 0.0],

    # 室内会议场景 - 光线稳定，需要清晰度和低噪点
    '室内会议_快速': [{'builtin_focus': 0.6, 'builtin_noise': 0.4}, 0.15],
    '室内会议_精细': [{'ericasz_useraes': 0.4, 'builtin_focus': 0.3, 'builtin_noise': 0.2, 'builtin_color': 0.1}, 0.25],
    
    # 室外活动场景 - 光线变化大，需要考虑亮度和动态范围
    '室外活动_运动强_快速': [{'builtin_focus': 0.6, 'builtin_lum': 0.4}, 0.2],
    '室外活动_运动强_精细': [{'ericasz_useraes': 0.35, 'builtin_focus': 0.25, 'builtin_lum': 0.2, 'builtin_noise': 0.1, 'builtin_color': 0.1}, 0.3],
    '室外活动_运动弱_快速': [{'builtin_lum': 0.5, 'builtin_focus': 0.5}, 0.15],
    '室外活动_运动弱_精细': [{'ericasz_useraes': 0.3, 'builtin_lum': 0.25, 'builtin_focus': 0.2, 'builtin_noise': 0.15, 'builtin_color': 0.1}, 0.25],
    
    # 体育比赛场景 - 高速运动，对焦和清晰度最重要
    '体育比赛_快速': [{'builtin_focus': 0.7, 'builtin_noise': 0.3}, 0.25],
    '体育比赛_精细': [{'ericasz_useraes': 0.3, 'builtin_focus': 0.4, 'builtin_noise': 0.15, 'builtin_lum': 0.15}, 0.35],
    
    # 文艺演出场景 - 光线复杂，色彩丰富，需要综合评估
    '文艺演出_舞台_快速': [{'builtin_lum': 0.5, 'builtin_color': 0.5}, 0.2],
    '文艺演出_舞台_精细': [{'ericasz_useraes': 0.35, 'builtin_lum': 0.2, 'builtin_focus': 0.15, 'builtin_color': 0.2, 'builtin_noise': 0.1}, 0.3],
    
    # 毕业典礼场景 - 重要场合，需要高质量选片
    '毕业典礼_快速': [{'builtin_focus': 0.5, 'builtin_lum': 0.5}, 0.15],
    '毕业典礼_精细': [{'ericasz_useraes': 0.4, 'builtin_focus': 0.25, 'builtin_lum': 0.15, 'builtin_noise': 0.1, 'builtin_color': 0.1}, 0.25],
    
    # 校园风景场景 - 静态拍摄，注重构图、色彩和画质
    '校园风景_快速': [{'builtin_color': 0.6, 'builtin_lum': 0.4}, 0.1],
    '校园风景_精细': [{'ericasz_useraes': 0.35, 'builtin_color': 0.25, 'builtin_lum': 0.2, 'builtin_focus': 0.1, 'builtin_noise': 0.1}, 0.2],
    
    # 采访场景 - 人像为主，需要清晰的面部
    '采访_快速': [{'builtin_focus': 0.6, 'builtin_noise': 0.4}, 0.15],
    '采访_精细': [{'ericasz_useraes': 0.35, 'builtin_focus': 0.3, 'builtin_noise': 0.2, 'builtin_lum': 0.15}, 0.25],
    
    # 讲座演讲场景 - 室内光线，主讲人清晰度高
    '讲座_快速': [{'builtin_focus': 0.6, 'builtin_lum': 0.4}, 0.15],
    '讲座_精细': [{'ericasz_useraes': 0.3, 'builtin_focus': 0.3, 'builtin_lum': 0.2, 'builtin_noise': 0.1, 'builtin_color': 0.1}, 0.2],
    
    # 社团活动场景 - 色彩丰富，气氛活跃
    '社团活动_快速': [{'builtin_color': 0.6, 'builtin_focus': 0.4}, 0.2],
    '社团活动_精细': [{'ericasz_useraes': 0.3, 'builtin_color': 0.25, 'builtin_focus': 0.2, 'builtin_lum': 0.15, 'builtin_noise': 0.1}, 0.25],
    
    # 学术报告场景 - 简洁清晰，PPT内容可读性
    '学术报告_快速': [{'builtin_focus': 0.6, 'builtin_lum': 0.4}, 0.15],
    '学术报告_精细': [{'ericasz_useraes': 0.35, 'builtin_focus': 0.25, 'builtin_lum': 0.2, 'builtin_noise': 0.1, 'builtin_color': 0.1}, 0.2],
    
    # 校园建筑场景 - 静态风景，注重线条和色彩
    '校园建筑_快速': [{'builtin_focus': 0.5, 'builtin_color': 0.5}, 0.1],
    '校园建筑_精细': [{'ericasz_useraes': 0.3, 'builtin_focus': 0.25, 'builtin_color': 0.2, 'builtin_lum': 0.15, 'builtin_noise': 0.1}, 0.15],
}

print(f'欢迎使用 AutoCull [版本 {__version__}] by Ericas Z.')
print('创建临时选片脚本文件 ...')
ac_files = sorted([f for f in os.listdir('__cull__') if f.endswith('.ac.py')])  # 收集并排序.ac.py文件
with open('cull_temp.py', 'w', encoding='utf-8', newline='') as outfile:
    for filename in ac_files:
        file_path = os.path.join('__cull__', filename)
        try:  # 合并文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                content = infile.read()
                clean_content = content.replace('\x1a', '')  # 移除U+001A字符
                outfile.write(clean_content)
                if not clean_content.endswith('\n'):
                    outfile.write('\n')
                outfile.write('\n')  # 文件间添加空行
        except Exception as e:
            print(f'跳过文件 {filename}: {e}')
print(f'合并完成！共 {len(ac_files)} 个文件')
print('读取选片函数 ...')
cullTemp = importlib.util.spec_from_file_location('cull_temp', 'cull_temp.py')
cullTemp_module = importlib.util.module_from_spec(cullTemp)
cullTemp.loader.exec_module(cullTemp_module)
funcs = [func for name, func in vars(cullTemp_module).items() if callable(func) and not name.startswith('_')]
funcs_names = [func.__name__ for func in funcs]
print(f'读取的选片函数：{len(funcs_names)} 个。')


class CullFuncMissingError(Exception):
    """当需要的选片函数缺失时抛出的异常"""
    pass


def calcScoreByPreset(preset: dict, image: Image.Image) -> list[float | list[float]] | list[str]:
    """通过给定的预设，计算传入图片的评分。
    :param preset: 预设。键为所调用选片函数的名称，值为对应权重（0~1）。例如 {'builtin_lum': 0.4, 'aes_aes': 0.6}
    :param image: PIL.Image.Image 格式的图片。
    :return: 列表，其第一项为加权计算后的图像综合评分，第二项为包含各分项评分（未乘各自权重）的列表。如有部分需要的选片函数不可用，则返回所有不可用函数名的列表。
    """
    global funcs_names, funcs
    if not all(func in funcs_names for func in preset.keys()): 
        return [func for func in preset.keys() if func not in funcs_names]
    used_funcs = [funcs[funcs_names.index(list(preset.keys())[i])] for i in range(len(preset))]
    scores = [func(image) for func in used_funcs]
    return [round(sum(score * weight for score, weight in zip(scores, preset.values())), 4), scores]


def batch_calc_scores(image_paths, preset_dict):
    """批量计算图片评分。
    :param image_paths: 包含大量图片绝对路径的列表
    :param preset_dict: 选片预设字典
    :return: 字典，键为图片绝对路径，值为函数计算得出的包含总评分和分项评分的列表
    :raises CullFuncMissingError: 当存在丢失的选片函数时
    """
    if not image_paths:
        return {}
    print(f"开始批量处理 {len(image_paths)} 张图片...")
    start_time = time()
    print("检查选片函数可用性...")
    first_image = Image.open(image_paths[0])
    check_result = calcScoreByPreset(preset_dict, first_image)  # 先检查第一张图片，确认所需函数是否存在
    if isinstance(check_result, list) and all(isinstance(item, str) for item in check_result):  # 返回的是缺失的函数名列表
        print(f"发现缺失的选片函数: {check_result}")
        raise CullFuncMissingError(f"Missing culling functions: {check_result}")

    results = {}
    processed_count = 0
    
    def process_image(image_path):
        nonlocal processed_count
        image_start_time = time()
        image = Image.open(image_path)
        result = calcScoreByPreset(preset_dict, image)
        processed_count += 1
        image_end_time = time()
        # 计算分辨率（百万像素）
        width, height = image.size
        megapixels = (width * height) / 1_000_000
        print(f"({processed_count}/{len(image_paths)}) 处理完成: {os.path.basename(image_path)} "
              f"| 分辨率: {megapixels:.2f}MP | 用时: {image_end_time - image_start_time:.2f}秒")
        return result
    
    # 使用线程池执行器进行多线程处理
    print(f"使用线程池处理 {len(image_paths)} 张图片...")
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(process_image, path): path for path in image_paths}  # 提交所有任务
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()  # 收集结果
                results[path] = result
            except Exception as e:
                print(f"处理图片 {path} 时出错: {e}")
    
    end_time = time()
    total_time = end_time - start_time
    print(f"批量处理完成! 总用时: {total_time:.2f}秒, 平均每张图片: {total_time/len(image_paths):.2f}秒")
    return results


class AutoCullHTTPRequestHandler(BaseHTTPRequestHandler):
    uploaded_images = []  # 存储上传的图片路径
    temp_dirs = []  # 用于跟踪创建的临时目录
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_file('index.html', 'text/html')
        elif self.path == '/api/presets':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(presets).encode())
        elif self.path.startswith('/api/thumbnail/'):
            self.serve_thumbnail()
        elif self.path == '/logs':
            self.handle_log_request()
        else:
            file_path = self.path.lstrip('/')
            if os.path.exists(file_path) and os.path.isfile(file_path):  # 尝试提供静态文件
                self.serve_file(file_path)
            else:
                self.send_error(404, "File not found")
                
    def handle_log_request(self):
        """处理日志请求"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        log_content = '\n'.join(log_messages)
        self.wfile.write(log_content.encode('utf-8'))  # 发送累积的日志消息
        
    def do_POST(self):
        match self.path:
            case '/api/import':
                self.handle_import()
            case '/api/cull':
                self.handle_cull()
            case '/api/export':
                self.handle_export()
            case _:
                self.send_error(404, "API endpoint not found")
            
    def serve_file(self, file_path, content_type=None):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if content_type is None:
                content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")

    def serve_thumbnail(self):
        try:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(self.path)  # 解析URL获取图片路径和查询参数
            path_parts = parsed_url.path.split('/', 4)  # 提取并解码图片路径部分
            if len(path_parts) < 4:
                self.send_error(400, "Bad Request")
                return

            encoded_path = path_parts[3]
            image_path = urllib.parse.unquote(encoded_path)  # 获取图片路径

            img = Image.open(image_path)  # 打开原始图片

            img.thumbnail((300, 300))  # 调整大小为短边300像素

            buffer = BytesIO()
            img.save(buffer, format='JPEG')  # 将图片保存到内存中
            buffer.seek(0)

            self.send_response(200)  # 发送响应
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(buffer.getvalue())
            
        except Exception as e:
            self.send_error(500, f"Error serving thumbnail: {str(e)}")

    def handle_import(self):
        content_type = self.headers['Content-Type']  # 解析 multipart/form-data
        if not content_type.startswith('multipart/form-data'):
            self.send_error(400, "Expected multipart/form-data")
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)  # 读取请求体

        boundary = content_type.split('boundary=')[1].encode()  # 简化的 multipart 解析
        parts = post_data.split(b'--' + boundary)
        
        saved_files = []
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)  # 记录临时目录以便清理
        print(f"创建临时目录: {temp_dir}")
        
        for part in parts:
            if b'filename="' in part:
                # 提取文件名
                filename_start = part.find(b'filename="') + len(b'filename="')
                filename_end = part.find(b'"', filename_start)
                filename = part[filename_start:filename_end].decode()

                content_start = part.find(b'\r\n\r\n') + 4
                content = part[content_start:]  # 提取文件内容
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(content)  # 保存到临时文件
                saved_files.append({
                    'path': file_path,
                    'filename': filename
                })
                print(f"保存文件: {filename} -> {file_path}")

        for file_info in saved_files:
            self.uploaded_images.append(file_info['path'])  # 添加到全局图片列表

        response_data = {
            'success': True,
            'images': [],
            'temp_dir': temp_dir
        }

        print(f"处理 {len(saved_files)} 个文件的元数据...")
        for file_info in saved_files:
            try:
                img = Image.open(file_info['path'])
                
                response_data['images'].append({
                    'path': file_info['path'],
                    'filename': file_info['filename'],
                    'width': img.width,
                    'height': img.height,
                    'filesize': f"{os.path.getsize(file_info['path']) / 1024:.1f} KB"
                })
                print(f"处理元数据完成: {file_info['filename']} ({img.width}x{img.height})")
            except Exception as e:
                print(f"处理图片 {file_info['filename']} 元数据时出错: {e}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())
        print(f"导入完成，共处理 {len(saved_files)} 个文件")

    def handle_cull(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        preset_key = data.get('preset_key')
        image_paths = data.get('image_paths', [])
        force_no_delete = data.get('force_no_delete', False)  # 获取强制不删除参数
        
        if preset_key not in presets:
            self.send_error(400, "Invalid preset key")
            return

        preset_config = presets[preset_key]
        preset_dict = preset_config[0]  # 选片函数字典
        delete_ratio = preset_config[1]  # 末位删除比例

        if force_no_delete:  # 如果勾选了强制不删除，则将删除比例设为0
            delete_ratio = 0
            print(f"[强制模式] 已禁用自动删除功能")
        
        print(f"开始使用预设 {preset_key} 处理 {len(image_paths)} 张图片...")
        print(f"预设权重: {preset_dict}")
        print(f"删除比例: {delete_ratio*100:.1f}%")
        
        try:
            results = batch_calc_scores(image_paths, preset_dict)
            preset_functions = list(preset_dict.keys())  # 获取当前预设的函数名称列表
            response_data = {
                'success': True,
                'images': [],
                'deleted_count': 0,
                'exported_folder': None
            }
            
            print(f"处理结果数据...")
            scored_images = []
            for path in image_paths:
                try:
                    img = Image.open(path)
                    filename = os.path.basename(path)
                    
                    image_info = {
                        'path': path,
                        'filename': filename,
                        'width': img.width,
                        'height': img.height,
                        'filesize': f"{os.path.getsize(path) / 1024:.1f} KB"
                    }
                    
                    if path in results and isinstance(results[path], list) and len(results[path]) >= 2:
                        image_info['score'] = results[path][0]
                        image_info['scores'] = results[path][1]
                        # 提取函数名称中最后一个下划线后的内容作为显示名称
                        image_info['score_names'] = [func_name.split('_')[-1] for func_name in preset_functions]
                        print(f"评分完成: {filename} -> 总分: {results[path][0]:.4f}")
                        scored_images.append((path, image_info))
                    else:
                        print(f"未评分: {filename}")
                        scored_images.append((path, image_info))
                    
                    response_data['images'].append(image_info)
                except Exception as e:
                    print(f"处理图片 {path} 时出错: {e}")

            if delete_ratio > 0 and len(scored_images) > 0:  # 如果设置了删除比例，则执行删除操作
                print(f"\n开始按分数排序并删除末位 {delete_ratio*100:.1f}% 的图片...")

                def sort_key(item):  # 按分数从低到高排序（有分数的优先）
                    path, info = item
                    return info.get('score', -1)  # 没有分数的排在最前面
                scored_images.sort(key=sort_key)

                delete_count = int(len(scored_images) * delete_ratio)  # 计算要删除的数量
                delete_count = min(delete_count, len(scored_images))  # 确保不超过总数
                
                if delete_count > 0:
                    images_to_delete = scored_images[:delete_count]  # 删除末位（分数最低）的图片
                    images_to_keep = scored_images[delete_count:]
                    
                    print(f"将删除 {delete_count} 张低分图片:")
                    deleted_paths = []
                    for path, info in images_to_delete:
                        print(f"  - {info['filename']} (分数: {info.get('score', 'N/A')})")
                        deleted_paths.append(path)

                    for path in deleted_paths:
                        if path in AutoCullHTTPRequestHandler.uploaded_images:  # 从 uploaded_images 中移除已删除的图片
                            AutoCullHTTPRequestHandler.uploaded_images.remove(path)

                    response_data['images'] = [info for _, info in images_to_keep]  # 更新响应数据，只保留未被删除的图片
                    response_data['deleted_count'] = delete_count
                    print(f"删除完成，剩余 {len(images_to_keep)} 张图片\n")

                    if images_to_keep:
                        print("开始自动导出保留的图片...")
                        try:
                            output_folder = os.path.join(os.getcwd(), 'output', str(int(time())))
                            os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
                            print(f"创建输出目录: {output_folder}")
                            for i, (path, info) in enumerate(images_to_keep):
                                filename = os.path.basename(path)
                                dest_path = os.path.join(output_folder, filename)
                                shutil.copy2(path, dest_path)  # 复制图片到输出文件夹
                                print(f"({i+1}/{len(images_to_keep)}) 导出: {filename}")
                            
                            response_data['exported_folder'] = output_folder
                            print(f"导出完成，文件保存在: {output_folder}\n")
                            
                        except Exception as e:
                            print(f"导出过程中出现错误: {e}")
                            response_data['export_error'] = str(e)
                else:
                    print("删除数量为0，跳过删除操作\n")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            print(f"选片处理完成，返回 {len(response_data['images'])} 个结果")
            
        except CullFuncMissingError as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode())
            print(f"选片函数缺失错误: {e}")
        except Exception as e:
            self.send_error(500, f"Error during culling: {str(e)}")
            print(f"选片处理过程中出现错误: {e}")

    def handle_export(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        image_paths = data.get('image_paths', [])
        
        if not image_paths:
            self.send_error(400, "No images to export")
            return
        
        print(f"开始导出 {len(image_paths)} 张图片...")
        try:
            output_folder = os.path.join(os.getcwd(), 'output', str(int(time())))
            os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
            print(f"创建输出目录: {output_folder}")
            for i, path in enumerate(image_paths):
                filename = os.path.basename(path)
                dest_path = os.path.join(output_folder, filename)
                shutil.copy2(path, dest_path)  # 复制图片到输出文件夹
                print(f"({i+1}/{len(image_paths)}) 导出: {filename}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True,
                'output_folder': output_folder
            }).encode())
            print(f"导出完成，文件保存在: {output_folder}")
            
        except Exception as e:
            self.send_error(500, f"Error during export: {str(e)}")
            print(f"导出过程中出现错误: {e}")


def run_server():
    server_address = ('127.0.0.1', 6789)
    httpd = HTTPServer(server_address, AutoCullHTTPRequestHandler)
    print(f"服务器启动，访问地址: http://{server_address[0]}:{server_address[1]}")
    print("等待客户端连接...")
    httpd.serve_forever()


if __name__ == '__main__':
    print("AutoCull Web UI 服务器正在启动...")
    run_server()
