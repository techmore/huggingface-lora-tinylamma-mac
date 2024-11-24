from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import os
from datetime import datetime, timedelta
import subprocess
import signal
import sys
import psutil
import threading
import time

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
connected_clients = set()
training_process = None
background_task = None

def analyze_training_data():
    try:
        with open('training_data.json', 'r') as f:
            data = json.load(f)
        
        # Basic statistics
        total_examples = len(data)
        avg_input_length = sum(len(example['input'].split()) for example in data) / total_examples
        avg_output_length = sum(len(example['output'].split()) for example in data) / total_examples
        
        # Extract topics (from instructions)
        topics = set()
        for example in data:
            # Simple topic extraction - split by common separators and take first significant word
            words = example['instruction'].lower().replace('?', ' ').replace('.', ' ').split()
            significant_words = [w for w in words if len(w) > 3 and w not in {'what', 'how', 'why', 'when', 'where', 'who', 'the', 'and', 'that', 'this'}]
            if significant_words:
                topics.add(significant_words[0])
        
        # Select a diverse sample of examples
        sample_size = min(10, total_examples)
        step = max(1, total_examples // sample_size)
        sample_examples = data[::step]
        
        return {
            'total_examples': total_examples,
            'avg_input_length': round(avg_input_length, 1),
            'avg_output_length': round(avg_output_length, 1),
            'unique_topics': len(topics),
            'examples': sample_examples
        }
    except Exception as e:
        print(f"Error analyzing training data: {e}")
        return {
            'total_examples': 0,
            'avg_input_length': 0,
            'avg_output_length': 0,
            'unique_topics': 0,
            'examples': []
        }

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds is None or seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{round(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def get_training_stats():
    """Get current training statistics with standardized metrics."""
    try:
        if os.path.exists('training_state.json'):
            try:
                with open('training_state.json', 'r') as f:
                    stats = json.load(f)
                
                # Verify training process is still running
                if training_process and training_process.poll() is None:
                    stats['status'] = 'Running'
                else:
                    stats['status'] = 'Stopped'
                
                # Emit training status to keep UI in sync
                for client_id in connected_clients:
                    socketio.emit('training_status', {'status': stats['status']}, room=client_id)
                
                # Update last_update time
                stats['last_update'] = datetime.now().strftime('%H:%M:%S')
                
                # Calculate remaining time
                if stats['samples_per_second'] > 0:
                    remaining_samples = stats['total_samples'] - stats['samples_processed']
                    stats['eta_seconds'] = remaining_samples / stats['samples_per_second']
                
                return stats
            except json.JSONDecodeError:
                print("Error reading training state file")
        
        # Return default stats if file doesn't exist or is corrupted
        return {
            'model_size_mb': 0.0,
            'epochs_completed': 0,
            'current_loss': 0.0,
            'samples_processed': 0,
            'total_samples': 0,
            'progress_percent': 0.0,
            'samples_per_second': 0.0,
            'elapsed_time_seconds': 0.0,
            'status': 'not started',
            'memory_usage_mb': 0.0,
            'last_update': datetime.now().strftime('%H:%M:%S'),
            'eta_seconds': 0.0
        }
    except Exception as e:
        print(f"Error getting training stats: {e}")
        return None

def emit_stats_update():
    """Emit stats update to all connected clients."""
    try:
        stats = get_training_stats()
        if stats:
            for client_id in connected_clients:
                socketio.emit('stats_update', stats, room=client_id)
    except Exception as e:
        print(f"Error in emit_stats_update: {e}")

def background_stats_update():
    """Background task to emit stats updates."""
    while True:
        try:
            emit_stats_update()
            # Sleep for slightly less than the training update interval (50 batches)
            # This ensures we don't miss updates while avoiding excessive polling
            time.sleep(4)  # 4 seconds is good balance for 50-batch interval
        except Exception as e:
            print(f"Error in background stats update: {e}")
            time.sleep(1)  # Sleep briefly on error before retrying

def start_training():
    """Start the training process."""
    global training_process, background_task
    
    try:
        if training_process is None or training_process.poll() is not None:
            # Kill any existing training processes
            try:
                os.system("pkill -f train_mlx.py")
                time.sleep(1)  # Wait for process to clean up
            except:
                pass
                
            # Start training process with environment variables
            env = os.environ.copy()
            env['MLX_USE_METAL'] = '1'  # Enable Metal backend
            env['MLX_METAL_MEMORY_LIMIT'] = '8192'  # 8GB memory limit
            
            # Start training process
            training_process = subprocess.Popen(
                [sys.executable, 'train_mlx.py'],
                env=env,
                stdout=None,
                stderr=None,
                universal_newlines=True,
                bufsize=1
            )
            print("Started training process")
            
            # Start background task if not already running
            if background_task is None:
                background_task = socketio.start_background_task(background_stats_update)
                print("Started background monitoring task")
                
    except Exception as e:
        print(f"Error in start_training: {e}")

def cleanup():
    """Clean up processes and tasks."""
    global training_process, background_task
    
    try:
        if training_process and training_process.poll() is None:
            print("Cleaning up training process...")
            training_process.terminate()
            training_process.wait(timeout=5)
            training_process = None
            
        background_task = None
        
    except Exception as e:
        print(f"Error in cleanup: {e}")

def get_latest_metrics():
    try:
        with open('training_metrics.jsonl', 'r') as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
    except:
        pass
    return None

@socketio.on('request_metrics')
def handle_metrics_request():
    metrics = get_latest_metrics()
    if metrics:
        emit('training_metrics', metrics)

@app.route('/')
def index():
    return render_template('monitor.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    try:
        client_id = request.sid
        connected_clients.add(client_id)
        print(f"Client {client_id} connected")
        
        # Start training if not already running
        if not training_process or training_process.poll() is not None:
            start_training()
            
        # Emit initial stats
        emit_stats_update()
        
    except Exception as e:
        print(f"Error in handle_connect: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    try:
        client_id = request.sid
        connected_clients.discard(client_id)
        print(f"Client {client_id} disconnected")
        
        # Stop background task if no clients connected
        if not connected_clients and background_task:
            cleanup()
            
    except Exception as e:
        print(f"Error in handle_disconnect: {e}")

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update requests from clients"""
    emit_stats_update()

@socketio.on('request_data_insights')
def handle_data_insights_request():
    insights = analyze_training_data()
    for client_id in connected_clients:
        socketio.emit('data_insights_update', insights, room=client_id)

@socketio.on('toggle_training')
def handle_toggle_training(data):
    global training_process
    try:
        if data.get('paused'):
            if training_process and training_process.poll() is None:
                # Send SIGSTOP to pause
                os.kill(training_process.pid, signal.SIGSTOP)
                status = 'Paused'
        else:
            if training_process and training_process.poll() is None:
                # Send SIGCONT to resume
                os.kill(training_process.pid, signal.SIGCONT)
                status = 'Running'
            else:
                # Start new training if not running
                start_training()
                status = 'Running'
        
        # Update all status indicators
        for client_id in connected_clients:
            socketio.emit('training_status', {'status': status}, room=client_id)
            
    except Exception as e:
        print(f"Error in toggle_training: {e}")
        status = 'Error'
        for client_id in connected_clients:
            socketio.emit('training_status', {'status': status}, room=client_id)

@socketio.on('save_checkpoint')
def handle_save_checkpoint():
    if training_process and training_process.poll() is None:
        # Create flag file for saving
        with open('save_checkpoint', 'w') as f:
            f.write('save')
        
        # Wait a bit and then emit the saved event
        for client_id in connected_clients:
            socketio.emit('checkpoint_saved', {'status': 'success'}, room=client_id)

@socketio.on('test_model')
def handle_test_model(data):
    try:
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return {'error': 'Please provide a prompt'}
            
        temperature = float(data.get('temperature', 0.7))
        max_length = int(data.get('max_length', 500))
        
        if not (0 <= temperature <= 1):
            return {'error': 'Temperature must be between 0 and 1'}
            
        if not (50 <= max_length <= 2000):
            return {'error': 'Max length must be between 50 and 2000'}

        # Load the model weights
        try:
            weights = mx.load("mlx_model_best.npz")
            model = load_model()
            model.update(tree_unflatten(list(weights.items())))
        except Exception as e:
            return {'error': f'Failed to load model: {str(e)}'}

        try:
            # Generate response
            response = generate_response(model, prompt, temperature=temperature, max_length=max_length)
            tokens_generated = len(response.split())
            
            return {
                'response': response,
                'tokens_generated': tokens_generated,
                'prompt': prompt
            }
        except Exception as e:
            return {'error': f'Failed to generate response: {str(e)}'}
            
    except ValueError as e:
        return {'error': f'Invalid parameter: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}

if __name__ == '__main__':
    # Register cleanup on exit
    import atexit
    atexit.register(cleanup)
    
    # Use eventlet for better async performance
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
