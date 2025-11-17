# This files looks at task videos from video dir and reorganize the files to kitchen | user | task | instance directory
import os
import shutil
from datetime import datetime, timedelta
import numpy as np
import argparse
import json

base_raw_dir = os.environ.get("BASE_RAW_DIR", "/Volumes/Autonomous2/EHP")
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="EHP04", help="Participant name")
parser.add_argument("--session_prefix", default="P24", help="Session prefix")
args = parser.parse_args()

participant = args.participant
session_prefix = args.session_prefix

#get the input data directory
x_data_dir = f"{base_raw_dir}/{participant}/"

# get the video and depth directories
base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"
base_thermal_vision_dir = f"{base_processed_dir}/{participant}/processed_thermal-vision_data"
base_thermal_audio_dir = f"{base_processed_dir}/{participant}/processed_thermal-audio_data"

# get all session information
watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
if not os.path.exists(watch_sessions_file):
    print(f"File {watch_sessions_file} does not exist. Exiting...")
    exit(1)

# get the watch files
with open(watch_sessions_file, "r") as f:
    watch_sessions = json.load(f)

# list all the collected sessions
def video_file_exists(base_filename, video_dir, depth_dir):
    # check if the file exists in the directory
    if os.path.exists(os.path.join(video_dir, base_filename)):
        return True
    if os.path.exists(os.path.join(depth_dir, base_filename)):
        return True
    return False


collected_sessions = {
        f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}": {
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4",base_video_dir, base_depth_dir)
    }

# loop over all files to find the session files
x_files = []
x_datetimes = []
for root, dirs, files in os.walk(x_data_dir):
    for x_file in files:
        if x_file.startswith("."):
            continue
        if os.path.getsize(os.path.join(root, x_file)) < 1000:
            continue

        try:
            # try to get the datetime from end of file
            base_filename = os.path.basename(x_file)
            if 'watch_recording' in base_filename:
                x_date, x_time = base_filename.split(".")[0].split("_")[-3:-1]
            else:      
                x_date, x_time = base_filename.split(".")[0].split("_")[-2:]
            x_datetime = datetime.strptime(f"{x_date}_{x_time}", "%Y%m%d_%H%M%S")
        except ValueError:
            print(f"File {x_file} does not have a valid date format. Skipping...")
            continue
        x_files.append(os.path.join(root, x_file))
        x_datetimes.append(x_datetime)

x_files = np.array(x_files)
x_datetimes = np.array(x_datetimes)
datetime_sort_idx = np.argsort(x_datetimes)
x_files = x_files[datetime_sort_idx]
x_datetimes = x_datetimes[datetime_sort_idx]


# the session output directory
x_reorganized_dir = f"{base_processed_dir}/{participant}/sessions/"
os.makedirs(x_reorganized_dir, exist_ok=True)
session_files = {}
for session_key, session_data in collected_sessions.items():
    # get the video and depth files
    session_prefix, session_idx, session_datetime = session_key.split("-")

    video_file = f"{base_video_dir}/{session_prefix}-{session_idx}-{session_data['start']}_{session_data['end']}.mp4"
    depth_file = f"{base_depth_dir}/{session_prefix}-{session_idx}-{session_data['start']}_{session_data['end']}.mp4"
    thermal_vision_file = f"{base_thermal_vision_dir}/{session_prefix}-{session_idx}-{session_data['start']}_{session_data['end']}.mp4"
    thermal_audio_file = f"{base_thermal_audio_dir}/{session_prefix}-{session_idx}-{session_data['start']}_{session_data['end']}.mp4"
    dominant_watch_files = session_data["files"].get("dominant", [])
    nondominant_watch_files = session_data["files"].get("nondominant", [])

    session_start_time = datetime.strptime(f"{session_data['start']}", "%Y%m%d_%H%M%S")
    session_end_time = datetime.strptime(f"{session_data['end']}", "%Y%m%d_%H%M%S")

    x_datetime_start = session_start_time - timedelta(minutes=45)
    x_datetime_end = session_end_time + timedelta(minutes=45)
    x_video_idx_start = np.where(x_datetimes >= x_datetime_start)[0][0]
    x_video_idx_end = np.where(x_datetimes < x_datetime_end)[0][-1]
    x_files_for_video = x_files[x_video_idx_start:x_video_idx_end]

    #skip watch files as they are already there
    x_files_for_video = [x_file for x_file in x_files_for_video if x_file not in dominant_watch_files and x_file not in nondominant_watch_files]

    if len(x_files_for_video) == 0:
        print(f"No x files found for session {session_key}. Skipping...")
        continue

    session_name = f"{session_prefix}-{session_idx}"
    session_dir = os.path.join(x_reorganized_dir, session_name)
    reorg_records = {}
    os.makedirs(session_dir, exist_ok=True)

    # copy combined video and depth file
    if os.path.exists(video_file):
        session_video_file = os.path.join(session_dir, f"{session_name}_video.mp4")
        if os.path.exists(session_video_file):
            print(f"File {session_video_file} already exists. Skipping...")
            reorg_records[session_video_file] = video_file
        shutil.copy2(video_file, session_video_file)
        reorg_records[session_video_file] = video_file

    if os.path.exists(depth_file):
        session_depth_file = os.path.join(session_dir, f"{session_name}_depth.mp4")
        if os.path.exists(session_depth_file):
            print(f"File {session_depth_file} already exists. Skipping...")
            reorg_records[session_depth_file] = depth_file
        shutil.copy2(depth_file, session_depth_file)
        reorg_records[session_depth_file] = depth_file

    if os.path.exists(thermal_vision_file):
        session_thermal_vision_file = os.path.join(session_dir, f"{session_name}_thermal-vision.mp4")
        if os.path.exists(session_thermal_vision_file):
            print(f"File {session_thermal_vision_file} already exists. Skipping...")
            reorg_records[session_thermal_vision_file] = thermal_vision_file
        shutil.copy2(thermal_vision_file, session_thermal_vision_file)
        reorg_records[session_thermal_vision_file] = thermal_vision_file

    if os.path.exists(thermal_audio_file):
        session_thermal_audio_file = os.path.join(session_dir, f"{session_name}_thermal-audio.mp4")
        if os.path.exists(session_thermal_audio_file):
            print(f"File {session_thermal_audio_file} already exists. Skipping...")
            reorg_records[session_thermal_audio_file] = thermal_audio_file
        shutil.copy2(thermal_audio_file, session_thermal_audio_file)
        reorg_records[session_thermal_audio_file] = thermal_audio_file

    # copy watch files with prefix "watch_dominant_" and "watch_nondominant_"
    for watch_file in dominant_watch_files:
        watch_file_path = os.path.join(x_data_dir, watch_file)
        watch_file_name = watch_file.split("/")[-1]
        watch_reorganized_path = os.path.join(session_dir, f"watch-dominant_{watch_file_name}")
        if not os.path.exists(watch_reorganized_path):
            shutil.copy2(watch_file_path, watch_reorganized_path)
            reorg_records[watch_reorganized_path] = watch_file_path
            print(f"Copying {watch_file} to {watch_reorganized_path}")
        else:
            print(f"File {watch_reorganized_path} already exists")
            reorg_records[watch_reorganized_path] = watch_file_path

    for watch_file in nondominant_watch_files:
        watch_file_path = os.path.join(x_data_dir, watch_file)
        watch_file_name = watch_file.split("/")[-1]
        watch_reorganized_path = os.path.join(session_dir, f"watch-nondominant_{watch_file_name}")
        if not os.path.exists(watch_reorganized_path):
            shutil.copy2(watch_file_path, watch_reorganized_path)
            reorg_records[watch_reorganized_path] = watch_file_path
            print(f"Copying {watch_file} to {watch_reorganized_path}")
        else:
            print(f"File {watch_reorganized_path} already exists")
            reorg_records[watch_reorganized_path] = watch_file_path



    # copy doppler and micarray files
    for x_file in x_files_for_video:
        x_file_path = os.path.join(x_data_dir, x_file)
        x_file_name = x_file.split("/")[-1]
        x_rig_name = x_file.split("/")[-3]
        x_dir_name = os.path.dirname(x_file_path)
        if "doppler" in x_file_name:
            x_reorganized_path = os.path.join(session_dir, f"{x_file_name}")
        elif "micarray" in x_file_name:
            x_reorganized_path = os.path.join(session_dir, f"{x_file_name}")
        elif "flir" in x_file_name:
            if 'audio/' in x_file_path:
                x_reorganized_path = os.path.join(session_dir, f"{x_file_name.replace('flir','flir_audio')}")
            else:
                x_reorganized_path = os.path.join(session_dir, f"{x_file_name.replace('flir','flir_vision')}")

        else:
            continue

        if not os.path.exists(x_reorganized_path):
            shutil.copy2(x_file_path, x_reorganized_path)
            reorg_records[x_reorganized_path] = x_file_path
            print(f"Copying {x_file} to {x_reorganized_path}")
        else:
            print(f"File {x_reorganized_path} already exists")
            reorg_records[x_reorganized_path] = x_file_path

    # save the reorg records
    time_curr = datetime.now().strftime("%Y%m%d_%H%M%S")
    reorg_records_file = os.path.join(session_dir, f"reorg_records_{time_curr}.json")
    json.dump(reorg_records, open(reorg_records_file, "w"), indent=4)
    print(f"Reorg records saved to {reorg_records_file}")
