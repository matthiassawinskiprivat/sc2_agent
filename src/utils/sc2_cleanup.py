import psutil
import subprocess
import os
from loguru import logger

def kill_switch_sc2():
    """
    Finds and kills all StarCraft II processes aggressively.
    """
    logger.info("Executing SC2 kill switch...")
    
    # 1. Use taskkill on Windows for maximum aggression
    if os.name == 'nt':
        try:
            # Kill by image name, /F is force, /T is tree (kill children)
            subprocess.run(["taskkill", "/F", "/IM", "SC2_x64.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "SC2.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "SC2_x86.exe", "/T"], capture_output=True)
        except Exception as e:
            logger.debug(f"Taskkill failed (probably no processes): {e}")

    # 2. Backup with psutil
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name']
            if name and any(x in name.upper() for x in ["SC2", "STARCRAFT"]):
                logger.info(f"Killing SC2 process: {name} (PID: {proc.info['pid']})")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} SC2 processes via psutil.")
    
    # 3. Clear burnysc2's internal tracking to prevent accumulation in logs/logic
    try:
        from sc2.sc2process import SC2Process
        SC2Process.procs = []
        logger.info("Cleared SC2Process.procs list.")
    except Exception as e:
        logger.debug(f"Could not clear SC2Process.procs: {e}")
        
    return killed_count


