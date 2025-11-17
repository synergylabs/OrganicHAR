# in this file, I wish to extract session for all users using the android watch data
import os
import json
import pandas as pd
from datetime import datetime

base_raw_dir = os.environ.get("BASE_RAW_DIR", "./raw_data")
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")

import argparse

parser = argparse.ArgumentParser(description="Process watch data for a participant.")
parser.add_argument("--participant", default="EHP01", help="Name of the participant data directory")
args = parser.parse_args()

participant = args.participant
participant_raw_dir = os.path.join(base_raw_dir, participant)
participant_processed_dir = os.path.join(base_processed_dir, participant)
os.makedirs(participant_processed_dir, exist_ok=True)

watch_suffixes = [
    "_acc.txt",
    "_gyro.txt",
    "_mag.txt",
    "_recording.mp4",
    "_audio.mp4",
    "_rotvec.txt",
]
watch_hand_locations = {
    "/watch/right/":"dominant",
    "/watch/left/":"non-dominant",
    "/watch/fe/":"dominant",
    "/watch/fossil/":"non-dominant",
}

user_watch_files = {
    'dominant': {},
    'non-dominant': {},
}

watch_sessions_raw_file = os.path.join(participant_processed_dir, "watch_sessions_raw.json")

if not os.path.exists(watch_sessions_raw_file):
    # Loop recursively through all subdirectories in the participant directory
    for root, dirs, files in os.walk(participant_raw_dir):
        for file in files:
            if any(file.endswith(suffix) for suffix in watch_suffixes):
                user_id = os.path.basename(root)
                curr_filepath = os.path.join(root, file)
                # filename is in format of <timestamp>_<suffix>. extract the timestamp from the filename and convert to datetime
                if 'watch_recording_' in file:
                    x_date, x_time = os.path.splitext(file)[0].split('watch_recording_')[-1].split('_')[:2]
                    file_ts = int(datetime.strptime(f"{x_date}_{x_time}", "%Y%m%d_%H%M%S").timestamp()*1000)
                    file_ts = str(file_ts)
                else:
                    file_ts = os.path.splitext(file)[0].split('_')[0]

                # if ("ana" in participant_raw_dir) or ("ricardo" in participant_raw_dir):
                #     # this timestamp is in UTC, localize it to UTC and then convert it to America/New_York timezone
                #     file_ts = pd.to_datetime(int(file_ts),unit='ms', utc=True).tz_convert('America/New_York')

                # classify the watch hand location based on the directory structure
                file_location = "dominant"  # default to dominant if no specific location is found
                for hand_location, location in watch_hand_locations.items():
                    if hand_location in curr_filepath:
                        file_location = location
                        break

                if file_ts not in user_watch_files[file_location]:
                    user_watch_files[file_location][file_ts] = []
                user_watch_files[file_location][file_ts].append(curr_filepath)

    # save the user_watch_files dictionary to a file for later use
    json.dump(user_watch_files, open(watch_sessions_raw_file, "w"), indent=4)

# Load the user_watch_files dictionary from the file
user_watch_files = json.load(open(watch_sessions_raw_file, "r"))

processed_session_file = os.path.join(participant_processed_dir, "watch_sessions.json")
if os.path.exists(processed_session_file):
    # If the processed session file already exists, load it to avoid reprocessing
    session_data = json.load(open(processed_session_file, "r"))
else:
    # not loop through the user_watch_files dictionary and create sessions
    session_data = {
    }
    # get the end time for all the sessions by looking at timestamp in last row of txt files

    for hand_location, files_dict in user_watch_files.items():
        for session_start_ts in sorted(files_dict.keys()):

            session_files = files_dict[session_start_ts]
            if len(session_files) == 0:
                continue

            end_ts = pd.to_datetime(int(session_start_ts),unit='ms')
            for sensor_file in session_files:
                if sensor_file.endswith('.mp4'):
                    # For video files, we can skip them as they don't have timestamps in the same way
                    continue
                elif sensor_file.endswith('rotvec.txt'):
                    df_sensor = pd.read_csv(sensor_file,  sep=",", names = ['watch_ts','epoch_ts','x','y','z','w','unnamed'], header=None)
                else:
                    df_sensor = pd.read_csv(sensor_file,  sep=",", names = ['watch_ts','epoch_ts','x','y','z'], header=None)
                if df_sensor.empty:
                    continue
                # get the last timestamp in the dataframe
                last_row_ts = df_sensor['epoch_ts'].iloc[-1]
                # if ("ana" in participant_raw_dir) or ("ricardo" in participant_raw_dir):
                #     # this timestamp is in UTC, localize it to UTC and then convert it to America/New_York timezone
                #     last_row_ts = pd.to_datetime(last_row_ts, unit='ms', utc=True).tz_convert('America/New_York')
                # else:
                last_row_ts = pd.to_datetime(last_row_ts, unit='ms')
                if last_row_ts > end_ts:
                    end_ts = last_row_ts
            # create a session entry
            session_start_ts = pd.to_datetime(int(session_start_ts),unit='ms')
            # if ("ana" in participant_raw_dir) or ("ricardo" in participant_raw_dir):
            #     # this timestamp is in UTC, localize it to UTC and then convert it to America/New_York timezone
            #     session_start_ts = session_start_ts.tz_localize('America/New_York')
            session_duration = (end_ts - session_start_ts).total_seconds()
            if session_duration <= 0:
                continue
            # get session key as start_timestamp and end time in %Y%m%d%H%M%S format
            session_key = f"{session_start_ts.strftime('%Y%m%d%H%M%S.%f')}_{end_ts.strftime('%Y%m%d%H%M%S.%f')}"
            if session_key not in session_data:
                session_data[session_key] = {
                    'start_time': session_start_ts.isoformat(),
                    'end_time': end_ts.isoformat(),
                    'duration_seconds': session_duration,
                    'files': {hand_location: session_files}
                }

    # Save the session data to a file
    json.dump(session_data, open(processed_session_file, "w"), indent=4)

merged_sessions_file = os.path.join(participant_processed_dir, "watch_sessions_merged.json")
if os.path.exists(merged_sessions_file):
    # If the merged session file already exists, load it to avoid reprocessing
    merged_session_data = json.load(open(merged_sessions_file, "r"))
else:
    # look at all session keys and merge sessions that overlap with each other
    merged_session_data = {}
    for session_key in sorted(session_data.keys()):
        session_info = session_data[session_key]
        session_start_ts = pd.to_datetime(session_info['start_time'])
        session_end_ts = pd.to_datetime(session_info['end_time'])

        # Check if this session overlaps with any existing merged sessions
        merged = False
        for merged_key in list(merged_session_data.keys()):
            merged_session_info = merged_session_data[merged_key]
            merged_start_ts = pd.to_datetime(merged_session_info['start_time'])
            merged_end_ts = pd.to_datetime(merged_session_info['end_time'])

            # Check for overlap
            if (session_start_ts <= merged_end_ts and session_end_ts >= merged_start_ts):
                # Merge the sessions
                new_start_ts = min(session_start_ts, merged_start_ts)
                new_end_ts = max(session_end_ts, merged_end_ts)
                new_duration_seconds = (new_end_ts - new_start_ts).total_seconds()
                # add all the keys from the files dictionary
                new_files = merged_session_info['files'].copy()
                new_files.update(session_info['files'])
                # Update the merged session data
                merged_session_data[merged_key] = {
                    'start_time': new_start_ts.isoformat(),
                    'end_time': new_end_ts.isoformat(),
                    'duration_seconds': new_duration_seconds,
                    'files': new_files
                }
                merged = True
                break

        if not merged:
            # No overlap found, add as a new session
            merged_session_data[session_key] = session_info.copy()

    # Save the merged session data to a file
    json.dump(merged_session_data, open(merged_sessions_file, "w"), indent=4)

# filter and order session data based on start time
filtered_sessions = []
min_duration = 60
max_duration = 10000
for session_key in sorted(merged_session_data.keys(), key=lambda k: pd.to_datetime(merged_session_data[k]['start_time'])):
    session_info = merged_session_data[session_key]
    session_start_ts = pd.to_datetime(session_info['start_time'])
    session_end_ts = pd.to_datetime(session_info['end_time'])
    # convert session start ts from UTC to America/New_York timezone
    session_start_ts = session_start_ts.tz_localize('UTC').tz_convert('Europe/Lisbon')
    session_end_ts = session_end_ts.tz_localize('UTC').tz_convert('Europe/Lisbon')
    session_duration = session_info['duration_seconds']

    if (session_duration >= min_duration) and (session_duration<=max_duration):
        filtered_sessions.append({
            'start': session_start_ts.strftime('%Y%m%d_%H%M%S'),
            'end': session_end_ts.strftime('%Y%m%d_%H%M%S'),
            'start_time': session_start_ts.isoformat(),
            'end_time': session_end_ts.isoformat(),
            'duration_seconds': session_duration,
            'files': session_info['files']
        })

# Save the filtered session data to a file
filtered_sessions_file = os.path.join(participant_processed_dir, "watch_sessions_filtered.json")
json.dump(filtered_sessions, open(filtered_sessions_file, "w"), indent=4)