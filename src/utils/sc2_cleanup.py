import psutil
from loguru import logger

def kill_switch_sc2():
    """
    Finds and kills all StarCraft II processes.
    """
    logger.info("Executing SC2 kill switch...")
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Check for SC2 process names (Windows and Linux)
            if proc.info['name'] in ["SC2_x64.exe", "SC2.exe", "SC2_x64", "SC2"]:
                logger.info(f"Killing SC2 process: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} SC2 processes.")
    else:
        logger.info("No SC2 processes found to kill.")
    return killed_count
