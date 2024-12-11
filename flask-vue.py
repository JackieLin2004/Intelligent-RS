from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from ShenDu.predict import predict_net,predict_chaofen,predict_color
from werkzeug.utils import secure_filename, send_from_directory

app = Flask(__name__)
# CORS(app)  # 允许跨域请求
CORS(app, resources={r"/*": {"origins": "*"}})

# 设置上传文件的保存路径
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # 允许上传的文件类型

# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<model_name>/<filename>')
def uploaded_file(model_name, filename):
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    return send_from_directory(upload_folder, filename)

@app.route('/results/<model_name>/<filename>', methods=['GET'])
def result_file(model_name, filename):
    try:
        result_folder = os.path.join(app.config['RESULT_FOLDER'], model_name)
        result_file_path = os.path.join(result_folder, filename)
        if os.path.exists(result_file_path):
            # 读取图片为二进制数据
            with open(result_file_path, 'rb') as f:
                image_data = f.read()
            # 返回图片数据流，设置正确的 MIME 类型
            return Response(image_data, mimetype='image/jpg')  # 根据实际图片类型设置 MIME
        else:
            return jsonify({'error': '文件未找到'}), 404
    except Exception as e:
        return jsonify({'error': '服务器内部错误', 'details': str(e)}), 500

@app.route('/fenlei_<model_name>', methods=['POST'])
def upload_file_fenlei(model_name):
    print(f"Received request for model: {model_name}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 如果没有选择文件
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 确保上传的是图片文件
    if file and allowed_file(file.filename):
        # 保存图片到uploads文件夹
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        img_path = os.path.join(upload_folder, filename)
        file.save(img_path)

        # 调用predict_google_net进行预测
        result_text, result_image, class_indict = predict_net(img_path, model_name)
        print(result_text)
        print(os.path.basename(result_image))

        return jsonify({
            'result_text': result_text,
            'result_image': f'/results/{model_name}/{os.path.basename(result_image)}'  # 返回可访问的URL
        })

    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/chaofen_<model_name>', methods=['POST'])
def upload_file_chaofen(model_name):
    print(f"Received request for model: {model_name}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 如果没有选择文件
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 确保上传的是图片文件
    if file and allowed_file(file.filename):
        # 保存图片到uploads文件夹
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        img_path = os.path.join(upload_folder, filename)
        file.save(img_path)

        # 调用predict_google_net进行预测
        result_image = predict_chaofen(img_path, model_name)
        print(os.path.basename(result_image))

        return jsonify({
            'result_text': ' ',
            'result_image': f'/results/{model_name}/{os.path.basename(result_image)}'  # 返回可访问的URL
        })

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/color_<model_name>', methods=['POST'])
def upload_file_color(model_name):
    print(f"Received request for model: {model_name}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 如果没有选择文件
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 确保上传的是图片文件
    if file and allowed_file(file.filename):
        # 保存图片到uploads文件夹
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        img_path = os.path.join(upload_folder, filename)
        file.save(img_path)

        # 调用predict_google_net进行预测
        result_image = predict_color(img_path, model_name)
        print(os.path.basename(result_image))

        return jsonify({
            'result_text': ' ',
            'result_image': f'/results/{model_name}/{os.path.basename(result_image)}'  # 返回可访问的URL
        })

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)