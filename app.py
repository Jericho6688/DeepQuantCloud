import os
import glob
import subprocess
import yaml
from flask import Flask, request, render_template, jsonify, Response, send_from_directory, url_for
import threading

app = Flask(__name__)

# Global generator and lock for backtesting
stream_generator_backtest = None
generator_lock_backtest = threading.Lock()

# Global generator and lock for data update
stream_generator_update = None
generator_lock_update = threading.Lock()

# Global generator and lock for prediction
stream_generator_prediction = None
generator_lock_prediction = threading.Lock()

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

# Get investment targets from investment_targets.yaml (for dropdown options)
@app.route("/get_targets")
def get_targets():
    try:
        with open("db2_create_table.yaml", "r") as f:
            targets = yaml.safe_load(f)
        return jsonify(targets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------
# New: Interface to list strategy chart files
@app.route("/get_strategy_charts")
def get_strategy_charts():
    try:
        # Find all files in the current directory that start with "strategy_chart_" and end with ".png"
        files = [f for f in os.listdir('.') if f.startswith("strategy_chart_") and f.endswith(".png")]
        files.sort()  # Sort by filename
        return jsonify({"strategy_charts": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Return strategy chart
@app.route("/image_strategy/<filename>")
def get_image_strategy(filename):
    return send_from_directory('.', filename)

# Return global capital chart (global_capital_curve.png)
@app.route("/image_global")
def get_image_global():
    return send_from_directory('.', 'global_capital_curve.png')

# Backtest related routes
@app.route("/run_backtest", methods=['POST'])
def run_backtest():
    global stream_generator_backtest
    with generator_lock_backtest:
        if stream_generator_backtest:
            return jsonify({'error': 'Backtest already running'}), 400

        try:
            # Delete old strategy chart files to avoid accumulating historical results
            old_files = glob.glob("strategy_chart_*.png")
            for file in old_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Failed to remove {file}: {e}")

            # Build configuration dictionary
            config = {
                'backtest': {
                    'start_date': request.form.get('backtest[start_date]', '2024-01-01'),
                    'end_date': request.form.get('backtest[end_date]', '2024-10-01'),
                    'cash': float(request.form.get('backtest[cash]', 100000.0)),
                    'commission': float(request.form.get('backtest[commission]', 0.001))
                },
                'strategies': []
            }

            # Parse strategy information
            for key, value in request.form.items():
                if key.startswith('strategies'):
                    parts = key.split('[')
                    if len(parts) < 3:
                        continue
                    try:
                        index = int(parts[1].split(']')[0])
                        field = parts[2].split(']')[0]
                        while len(config['strategies']) <= index:
                            config['strategies'].append({})
                        if field == 'percents':
                            config['strategies'][index][field] = int(value)
                        else:
                            config['strategies'][index][field] = value
                    except (ValueError, IndexError):
                        continue

            # Write configuration to YAML file
            with open("backtester1.yaml", "w") as f:
                yaml.dump(config, f)

            # Define backtest generator
            def generate_backtest():
                global stream_generator_backtest
                try:
                    process = subprocess.Popen(
                        ["python", "backtester1.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                        text=True
                    )
                    for line in iter(process.stdout.readline, ''):
                        yield f"data: {line}\n\n"
                    process.stdout.close()
                    return_code = process.wait()
                    if return_code:
                        yield f"data: Error: Backtest process exited with code {return_code}\n\n"
                    else:
                        yield "data: STREAM_CLOSED\n\n"
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
                finally:
                    with generator_lock_backtest:
                        stream_generator_backtest = None

            stream_generator_backtest = generate_backtest()
            return jsonify({'status': 'Backtest started'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route("/stream_backtest")
def stream_backtest():
    global stream_generator_backtest
    with generator_lock_backtest:
        if not stream_generator_backtest:
            return jsonify({'error': 'No backtest running'}), 400
        return Response(stream_generator_backtest, mimetype='text/event-stream')

@app.route("/image_backtest")
def get_image_backtest():
    return send_from_directory('.', 'backtester1_graph.png')

# --------------------------
# Data update related routes
@app.route("/run_update", methods=['POST'])
def run_update():
    global stream_generator_update
    with generator_lock_update:
        if stream_generator_update:
            return jsonify({'error': 'Data update already running'}), 400

        try:
            def generate_update():
                global stream_generator_update
                try:
                    process1 = subprocess.Popen(
                        ["python", "db4_update_table.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                        text=True
                    )
                    for line in iter(process1.stdout.readline, ''):
                        yield f"data: {line}\n\n"
                    process1.stdout.close()
                    return_code1 = process1.wait()
                    if return_code1:
                        yield f"data: Error: db4_update_table.py exited with code {return_code1}\n\n"
                    else:
                        yield "data: STREAM_CLOSED\n\n"
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
                finally:
                    with generator_lock_update:
                        stream_generator_update = None

            stream_generator_update = generate_update()
            return jsonify({'status': 'Data update started'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route("/stream_update")
def stream_update():
    global stream_generator_update
    with generator_lock_update:
        if not stream_generator_update:
            return jsonify({'error': 'No data update running'}), 400
        return Response(stream_generator_update, mimetype='text/event-stream')

# --------------------------
# Prediction related routes
@app.route("/run_prediction", methods=['POST'])
def run_prediction():
    global stream_generator_prediction
    with generator_lock_prediction:
        if stream_generator_prediction:
            return jsonify({'error': 'Prediction already running'}), 400

        try:
            target = request.form.get('target')
            if not target or '.' not in target:
                return jsonify({'error': 'Target is required and must be in the format schema.table'}), 400
            schema, table = target.split('.')
            if not os.path.exists("prediction.yaml"):
                return jsonify({'error': 'prediction.yaml not found'}), 400

            with open("prediction.yaml", "r") as f:
                config = yaml.safe_load(f)

            if 'prediction' not in config:
                config['prediction'] = {}
            config['prediction']['schema'] = schema
            config['prediction']['table'] = table

            with open("prediction.yaml", "w") as f:
                yaml.dump(config, f)

            def generate_prediction():
                global stream_generator_prediction
                try:
                    process = subprocess.Popen(
                        ["python", "prediction.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=os.getcwd(),
                        text=True
                    )
                    stdout, stderr = process.communicate()
                    if process.returncode:
                        error_message = stderr if stderr else "Unknown error."
                        yield f"data: Error: Prediction process exited with code {process.returncode}. {error_message}\n\n"
                    else:
                        yield "data: PREDICTION_COMPLETED\n\n"
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
                finally:
                    with generator_lock_prediction:
                        stream_generator_prediction = None

            stream_generator_prediction = generate_prediction()
            return jsonify({'status': 'Prediction started'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route("/stream_prediction")
def stream_prediction():
    global stream_generator_prediction
    with generator_lock_prediction:
        if not stream_generator_prediction:
            return jsonify({'error': 'No prediction running'}), 400
        return Response(stream_generator_prediction, mimetype='text/event-stream')

@app.route("/image_prediction")
def get_image_prediction():
    return send_from_directory('.', 'prediction_graph.png')

# Get all MP3 files in the static directory
@app.route("/get_music_files")
def get_music_files():
    static_dir = os.path.join(app.root_path, 'static')
    music_files = [f for f in os.listdir(static_dir) if f.lower().endswith('.mp3')]
    music_urls = [url_for('static', filename=f) for f in music_files]
    return jsonify({'music_files': music_urls})

# Get strategy scripts in the strategies directory
@app.route("/get_strategies")
def get_strategies():
    strategies_dir = os.path.join(app.root_path, 'strategies')
    if not os.path.isdir(strategies_dir):
        return jsonify({'error': 'Strategies directory not found'}), 400

    strategy_files = [
        f for f in os.listdir(strategies_dir)
        if f.endswith('.py') and f != '__init__.py'
    ]
    return jsonify({'strategies': strategy_files})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)