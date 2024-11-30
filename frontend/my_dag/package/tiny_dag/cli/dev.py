import os
import time
import threading
import webbrowser
from honcho.manager import Manager

def open_browser():
    """Open browser after a short delay with specific window size"""
    time.sleep(3)
    # On macOS, Chrome accepts JavaScript to set window size
    chrome_path = 'open -a "Google Chrome" %s --args --window-size=1200,800 --window-position=100,100'
    webbrowser.get(chrome_path).open('http://localhost:5173')

def run_dev():
    manager = Manager()
    
    # Start frontend first and launch browser
    manager.add_process(
        'frontend', 
        'cd src && npm install && npm run dev'
    )
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Add backend with initial output suppression
    manager.add_process(
        'backend', 
        'sleep 5 2>/dev/null && tiny_dag_backend'
    )
    
    os.environ['PYTHONUNBUFFERED'] = '1'
    manager.loop()

if __name__ == "__main__":
    run_dev()