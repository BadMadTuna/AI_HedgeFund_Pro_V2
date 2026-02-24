import os
import time
import subprocess

# --- CONFIG ---
IDLE_LIMIT_MINUTES = 20
CHECK_INTERVAL_SECONDS = 60

def is_user_connected():
    """Checks if there is an active web socket connection to Streamlit (Port 8501)."""
    try:
        # ss command looks for ESTABLISHED connections on port 8501
        result = subprocess.check_output("ss -tnp | grep :8501 | grep ESTAB", shell=True)
        return len(result.strip()) > 0
    except subprocess.CalledProcessError:
        # grep returns an error code if it finds nothing
        return False

def monitor():
    idle_minutes = 0
    print(f"Starting Auto-Shutdown Monitor. Idle limit: {IDLE_LIMIT_MINUTES} mins.")
    
    while True:
        if is_user_connected():
            idle_minutes = 0  # Reset timer if user is active
        else:
            idle_minutes += 1
            
        if idle_minutes >= IDLE_LIMIT_MINUTES:
            print(f"Server idle for {IDLE_LIMIT_MINUTES} minutes. Initiating shutdown...")
            # Safely power down the EC2 instance
            os.system("sudo shutdown -h now")
            break
            
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor()